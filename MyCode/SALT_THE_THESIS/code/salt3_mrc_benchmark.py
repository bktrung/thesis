"""MRC (extractive QA) benchmark engine — UIT-ViQuAD 2.0, answerable-only.

One harness, two model families through the same span-prediction pipeline:
  * pretrained HF encoders (PhoBERT / XLM-R / ViDeBERTa) via AutoModelForQuestionAnswering
  * our SALT NeoBERT milestone dirs via the NeoBERTQuestionAnswering head

Consolidates the proven offset-aligned span building + VnCoreNLP segmentation
(answer-span remap) + constrained-span SQuAD EM/F1 decode that previously lived
inline in notebook 03, so baselines and milestones are scored identically.
Word-segmentation is applied per model spec (PhoBERT/ViDeBERTa need it; XLM-R and
our syllable-tokenizer NeoBERT do not). Resumable: skips (model, seed) already
recorded in the shared results jsonl.
"""
from __future__ import annotations

import gc
import random
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import (
    AutoConfig, AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoTokenizer,
    DefaultDataCollator, EarlyStoppingCallback, Trainer, TrainingArguments,
)

from salt3_common import (
    NeoBERTQuestionAnswering, append_jsonl, ensure_dir, load_model_safe,
    maybe_segment, read_jsonl, set_seed, write_json,
)


@dataclass
class MRCConfig:
    """Full-data thesis-table defaults; caps=None -> use all rows."""
    max_len: int = 384
    lr: float = 2e-5
    epochs: int = 5
    batch_size: int = 8
    train_cap: int | None = None       # None -> full ViQuAD train (~18k answerable)
    eval_cap: int | None = None        # None -> full val/test
    patience: int = 2
    max_answer_len: int = 30
    seeds: tuple[int, ...] = (42, 123, 456)
    tmp_root: Path = field(default_factory=lambda: Path("/content/mrc_bench_tmp"))


def load_viquad_answerable(repo: str = "taidng/UIT-ViQuAD2.0", seed: int = 42):
    """-> dict(train, validation, test). Hub eval split halved into val/test.
    Answerable rows kept by the builder (is_impossible dropped there)."""
    from datasets import load_dataset

    ds = load_dataset(repo)
    train = ds["train"]
    ev = ds["validation"] if "validation" in ds else ds["test"]
    sp = ev.train_test_split(test_size=0.5, seed=seed)
    return {"train": train, "validation": sp["train"], "test": sp["test"]}


def _first_answer(ex):
    """Return (answer_text, char_start) for an answerable example, else None."""
    if ex.get("is_impossible", False):
        return None
    a = ex.get("answers")
    if not isinstance(a, dict) or not a.get("text"):
        return None
    t = a["text"][0]
    if t is None:
        return None
    t = str(t).strip()
    return (t, int(a["answer_start"][0])) if t else None


def _wordlevel_offsets(tokenizer, question, context, max_len):
    """Synthesize (input_ids, attention_mask, offsets, seq_ids) for a SLOW, word-level
    tokenizer (PhoBERT has no fast variant). Input must be whitespace-segmented; each
    word's subword tokens inherit that word's char span in `context`. Answers fall on
    word boundaries after segmentation, so this is sufficient for span alignment.
    only_second truncation: the context is trimmed to fit, the question is kept whole."""
    import re

    q_ids = tokenizer.encode(question, add_special_tokens=False)
    ctx_ids, ctx_offsets = [], []
    for m in re.finditer(r"\S+", context):
        for tid in tokenizer.encode(m.group(0), add_special_tokens=False):
            ctx_ids.append(tid)
            ctx_offsets.append((m.start(), m.end()))
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    max_ctx = max(0, max_len - len(q_ids) - n_special)
    ctx_ids, ctx_offsets = ctx_ids[:max_ctx], ctx_offsets[:max_ctx]

    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, ctx_ids)
    sp_mask = tokenizer.get_special_tokens_mask(q_ids, ctx_ids, already_has_special_tokens=False)
    seq_ids, offsets, qi, ci = [], [], 0, 0
    for pos, is_sp in enumerate(sp_mask):
        if is_sp:
            seq_ids.append(None); offsets.append((0, 0))
        elif qi < len(q_ids):
            seq_ids.append(0); offsets.append((0, 0)); qi += 1
        else:
            seq_ids.append(1); offsets.append(ctx_offsets[ci]); ci += 1
    attn = [1] * len(input_ids)
    pad = max_len - len(input_ids)
    if pad > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad
        attn += [0] * pad
        seq_ids += [None] * pad
        offsets += [(0, 0)] * pad
    return {"input_ids": input_ids, "attention_mask": attn}, offsets, seq_ids


def _encode_qa(tokenizer, question, context, max_len):
    """Fast tokenizer -> native offset_mapping/sequence_ids; slow -> word-level synth."""
    if tokenizer.is_fast:
        enc = tokenizer(question, context, truncation="only_second", max_length=max_len,
                        return_offsets_mapping=True, padding="max_length",
                        return_token_type_ids=False)
        return enc, enc.pop("offset_mapping"), enc.sequence_ids()
    return _wordlevel_offsets(tokenizer, question, context, max_len)


def build_qa_rows(split_ds, tokenizer, spec, cap, max_len, model_vocab=None, seed=42):
    """Tokenize (question, context), align char-span to token-span via offsets.
    For segment-spec models, word-segment q/context/answer first and re-find the
    answer span in the segmented context (offsets shift under underscore-joins).
    Slow word-level tokenizers (PhoBERT) require segment=True so context is
    whitespace-segmented for the synthetic-offset path."""
    from datasets import Dataset

    if not tokenizer.is_fast and not (spec and spec.get("segment")):
        print(f"  {type(tokenizer).__name__}: slow tokenizer needs segment=True, skipping MRC")
        return Dataset.from_list([])
    random.seed(seed)
    idxs = list(range(len(split_ds)))
    random.shuffle(idxs)
    rows, drops = [], {"no_answer": 0, "span_not_found": 0, "truncated": 0}
    for j in idxs:
        if cap and len(rows) >= cap:
            break
        ex = split_ds[j]
        fa = _first_answer(ex)
        if fa is None:
            drops["no_answer"] += 1
            continue
        text, char_start = fa
        char_end = char_start + len(text)
        question, context = ex["question"], ex["context"]
        if spec and spec.get("segment"):
            question = maybe_segment([question], spec)[0]
            ctx_seg = maybe_segment([context], spec)[0]
            txt_seg = maybe_segment([text], spec)[0]
            idx = ctx_seg.find(txt_seg)
            if idx < 0:
                drops["span_not_found"] += 1
                continue
            context, text, char_start, char_end = ctx_seg, txt_seg, idx, idx + len(txt_seg)
        enc, offsets, seq_ids = _encode_qa(tokenizer, question, context, max_len)
        ctx_mask = [int(seq_ids[ti] == 1) for ti in range(len(offsets))]
        start_t = end_t = None
        for ti, (cs, ce) in enumerate(offsets):
            if seq_ids[ti] != 1 or cs is None or ce is None:
                continue
            if cs <= char_start < ce and start_t is None:
                start_t = ti
            if cs < char_end <= ce:
                end_t = ti
        if start_t is None or end_t is None or end_t < start_t:
            ctx_end = max((ti for ti, sid in enumerate(seq_ids) if sid == 1), default=-1)
            drops["truncated" if (ctx_end >= 0 and char_end > offsets[ctx_end][1]) else "span_not_found"] += 1
            continue
        enc["start_positions"] = start_t
        enc["end_positions"] = end_t
        enc["context_token_mask"] = ctx_mask
        enc["example_id"] = str(ex.get("id", f"row{j}"))
        enc["answers"] = {"text": [text], "answer_start": [char_start]}
        rows.append(enc)
    if model_vocab and rows:
        before = len(rows)
        rows = [r for r in rows if max(r["input_ids"]) < model_vocab]
        if len(rows) < before:
            print(f"  filtered {before - len(rows)} OOB-vocab rows")
    print(f"  QA rows: {len(rows)} kept, drops={drops}")
    return Dataset.from_list(rows)


def squad_post_eval(model, tokenizer, eval_ds, device, batch_size=16, max_answer_len=30):
    """Predict spans constrained to context tokens; return SQuAD {f1, exact_match}."""
    import evaluate

    squad = evaluate.load("squad")
    model.eval()
    preds, refs = [], []
    for bs in range(0, len(eval_ds), batch_size):
        batch = [eval_ds[i] for i in range(bs, min(bs + batch_size, len(eval_ds)))]
        input_ids = torch.tensor([ex["input_ids"] for ex in batch]).to(device)
        attn = torch.tensor([ex["attention_mask"] for ex in batch]).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
        sl_b, el_b = out["start_logits"].float().cpu(), out["end_logits"].float().cpu()
        for bi, ex in enumerate(batch):
            sl, el = sl_b[bi], el_b[bi]
            valid = [k for k, ok in enumerate(ex.get("context_token_mask", [1] * len(ex["input_ids"]))) if ok]
            if not valid:
                preds.append({"prediction_text": "", "id": ex["example_id"]})
                refs.append({"id": ex["example_id"], "answers": ex["answers"]})
                continue
            best = None
            for k, si0 in enumerate(valid):
                for ei0 in valid[k:bisect_right(valid, si0 + max_answer_len)]:
                    score = float(sl[si0] + el[ei0])
                    if best is None or score > best[0]:
                        best = (score, si0, ei0)
            if best is None:  # no admissible span (e.g. max_answer_len<=0) -> empty prediction
                preds.append({"prediction_text": "", "id": ex["example_id"]})
                refs.append({"id": ex["example_id"], "answers": ex["answers"]})
                continue
            _, si, ei = best
            pred_text = tokenizer.decode(ex["input_ids"][si:ei + 1], skip_special_tokens=True)
            preds.append({"prediction_text": pred_text, "id": ex["example_id"]})
            refs.append({"id": ex["example_id"], "answers": ex["answers"]})
    return squad.compute(predictions=preds, references=refs)


def build_qa_model(spec, path, device):
    """Dispatch: NeoBERT milestone dir -> custom QA head; HF hub id -> AutoModelForQA."""
    if spec.get("kind") == "neobert":
        base = load_model_safe(path, model_cls=AutoModelForMaskedLM, device="cpu")
        return NeoBERTQuestionAnswering(base).to(device)
    return AutoModelForQuestionAnswering.from_pretrained(path, trust_remote_code=True).to(device)


def finetune_mrc(spec, splits, *, seed, cfg: MRCConfig, device="cuda"):
    """Fine-tune one model on ViQuAD; return test {f1, exact_match, n_params}."""
    set_seed(seed)
    path = spec["path"]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    if not tokenizer.is_fast and not spec.get("segment"):
        # slow tokenizers (PhoBERT) are supported only via the word-level offset path,
        # which needs segment=True; without it we cannot align spans -> skip loudly.
        print(f"  {spec['name']}: slow tokenizer without segment=True, skipping MRC")
        return None
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_vocab = getattr(config, "vocab_size", None)
    eff_max = min(cfg.max_len, max(64, getattr(config, "max_position_embeddings", 512) - 2))

    model = build_qa_model(spec, path, device)
    n_params = sum(p.numel() for p in model.parameters())

    tr = build_qa_rows(splits["train"], tokenizer, spec, cfg.train_cap, eff_max, model_vocab, seed)
    ev_val = build_qa_rows(splits["validation"], tokenizer, spec, cfg.eval_cap, eff_max, model_vocab, seed)
    ev_test = build_qa_rows(splits["test"], tokenizer, spec, cfg.eval_cap, eff_max, model_vocab, seed)
    if len(tr) < 50 or len(ev_val) < 20 or len(ev_test) < 10:
        print("  too few QA rows -> skip")
        del model
        gc.collect()
        return None

    keep = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    tr_clean = tr.remove_columns([c for c in tr.column_names if c not in keep])
    val_clean = ev_val.remove_columns([c for c in ev_val.column_names if c not in keep])

    out = ensure_dir(cfg.tmp_root / spec["name"].replace("/", "_") / f"seed{seed}")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size,
        weight_decay=0.01, warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="loss",
        greater_is_better=False, report_to="none", seed=seed, remove_unused_columns=False,
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16, tf32=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr_clean, eval_dataset=val_clean,
                      processing_class=tokenizer, data_collator=DefaultDataCollator(),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)])
    trainer.train()
    write_json(out / "training_log.json", trainer.state.log_history)
    metrics = squad_post_eval(model, tokenizer, ev_test, device, max_answer_len=cfg.max_answer_len)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"f1": float(metrics["f1"]), "exact_match": float(metrics["exact_match"]), "n_params": n_params}


def run_benchmark(specs, splits, results_jsonl, *, cfg: MRCConfig, task="MRC-ViQuAD",
                  device="cuda", force=False):
    """For each model spec x seed: fine-tune & append a result row. Skips done rows."""
    results_jsonl = Path(results_jsonl)
    done = read_jsonl(results_jsonl) if results_jsonl.exists() else []

    def is_done(name, seed):
        return any(r.get("task") == task and r.get("model") == name and r.get("seed") == seed
                   for r in done)

    for name, spec in specs.items():
        spec = {**spec, "name": name}
        for seed in cfg.seeds:
            if is_done(name, seed) and not force:
                print(f"  skip existing {task}/{name}/seed{seed}")
                continue
            print(f"  {task}: {name}/seed{seed} ...")
            m = finetune_mrc(spec, splits, seed=seed, cfg=cfg, device=device)
            if m is None:
                continue
            row = {"task": task, "model": name, "family": spec.get("family"),
                   "segment": bool(spec.get("segment")), "milestone_docs": spec.get("milestone_docs"),
                   "checkpoint_kind": spec.get("checkpoint_kind"), "seed": seed, **m}
            append_jsonl(results_jsonl, row)
            done.append(row)
            print(f"    -> f1={m['f1']:.2f} em={m['exact_match']:.2f}")
    return read_jsonl(results_jsonl)
