"""MRC (extractive QA) probe for the CPT bake-off — UIT-ViQuAD 2.0.

The hardest, most discriminative downstream probe: span prediction stresses the
encoder representation far more than sentence classification. Ports notebook 03's
proven offset-aligned span building + SQuAD EM/F1 post-eval, specialised to the
all-NeoBERT, model-dir case (no segmentation branch). Uses the QA head that
already lives in salt3_common (NeoBERTQuestionAnswering).
"""
from __future__ import annotations

import gc
import random
from bisect import bisect_right
from pathlib import Path

import torch
from transformers import (
    AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator,
    EarlyStoppingCallback, Trainer, TrainingArguments,
)

from salt3_common import (
    NeoBERTQuestionAnswering, append_jsonl, ensure_dir, load_model_safe,
    read_jsonl, set_seed, write_json,
)


def _load_fast_tokenizer(model_dir):
    """Return a FAST tokenizer (required for offset_mapping in QA). The artifacts
    pin tokenizer_class=DebertaV2Tokenizer (slow, vocab_type=spm) in
    tokenizer_config but ship a fast tokenizer.json with a proper [CLS] A [SEP] B
    [SEP] pair template — so load that directly when the default load is slow."""
    import json
    from transformers import PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    if tok.is_fast:
        return tok
    tj = Path(model_dir) / "tokenizer.json"
    if not tj.exists():
        return tok  # caller checks is_fast and skips MRC
    cfgp = Path(model_dir) / "tokenizer_config.json"
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    specials = {k: cfg[k] for k in ("bos_token", "eos_token", "unk_token", "sep_token",
                                    "pad_token", "cls_token", "mask_token") if cfg.get(k)}
    return PreTrainedTokenizerFast(tokenizer_file=str(tj), **specials)


def load_viquad(repo: str = "taidng/UIT-ViQuAD2.0", seed: int = 42):
    """-> dict(train, validation, test). The hub eval split is halved into val/test."""
    from datasets import load_dataset

    ds = load_dataset(repo)
    train = ds["train"]
    ev = ds["validation"] if "validation" in ds else ds["test"]
    sp = ev.train_test_split(test_size=0.5, seed=seed)
    return {"train": train, "validation": sp["train"], "test": sp["test"]}


def _first_answer(ex):
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


def _build_qa_rows(split_ds, tokenizer, cap, max_len, model_vocab=None, seed=42):
    """Tokenize (question, context), align char-span to token-span via offsets.
    Requires a fast tokenizer; drops no-answer / unalignable / truncated examples."""
    from datasets import Dataset

    if not tokenizer.is_fast:
        print(f"  {type(tokenizer).__name__}: slow tokenizer, skipping MRC")
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
        enc = tokenizer(ex["question"], ex["context"], truncation="only_second",
                        max_length=max_len, return_offsets_mapping=True,
                        padding="max_length", return_token_type_ids=False)
        offsets = enc.pop("offset_mapping")
        seq_ids = enc.sequence_ids()
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


def _squad_post_eval(model, tokenizer, eval_ds, device, batch_size=16, max_answer_len=30):
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
            valid = [j for j, ok in enumerate(ex.get("context_token_mask", [1] * len(ex["input_ids"]))) if ok]
            if not valid:
                pred_text = ""
            else:
                best = None
                for k, si0 in enumerate(valid):
                    for ei0 in valid[k:bisect_right(valid, si0 + max_answer_len)]:
                        score = float(sl[si0] + el[ei0])
                        if best is None or score > best[0]:
                            best = (score, si0, ei0)
                _, si, ei = best
                pred_text = tokenizer.decode(ex["input_ids"][si:ei + 1], skip_special_tokens=True)
            preds.append({"prediction_text": pred_text, "id": ex["example_id"]})
            refs.append({"id": ex["example_id"], "answers": ex["answers"]})
    return squad.compute(predictions=preds, references=refs)


def finetune_mrc(model_dir, qa_train, qa_val, qa_test, *, seed=42, max_len=384, lr=2e-5,
                 epochs=5, batch_size=8, train_cap=None, eval_cap=500, patience=2,
                 device="cuda"):
    """Fine-tune NeoBERTQuestionAnswering on ViQuAD; return test {f1, exact_match}."""
    set_seed(seed)
    tokenizer = _load_fast_tokenizer(model_dir)
    if not tokenizer.is_fast:
        print(f"  {model_dir}: no fast tokenizer available, skipping MRC")
        return None
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_vocab = getattr(config, "vocab_size", None)
    eff_max = min(max_len, max(64, getattr(config, "max_position_embeddings", 512) - 2))

    base = load_model_safe(model_dir, model_cls=AutoModelForMaskedLM, device="cpu")
    model = NeoBERTQuestionAnswering(base).to(device)

    tr = _build_qa_rows(qa_train, tokenizer, train_cap, eff_max, model_vocab, seed)
    ev_val = _build_qa_rows(qa_val, tokenizer, eval_cap, eff_max, model_vocab, seed)
    ev_test = _build_qa_rows(qa_test, tokenizer, eval_cap, eff_max, model_vocab, seed)
    if len(tr) < 50 or len(ev_val) < 20 or len(ev_test) < 10:
        print("  too few QA rows -> skip")
        del model, base
        gc.collect()
        return None

    train_only = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    tr_clean = tr.remove_columns([c for c in tr.column_names if c not in train_only])
    val_clean = ev_val.remove_columns([c for c in ev_val.column_names if c not in train_only])

    tag = f"{Path(model_dir).parent.name}__{Path(model_dir).name}"
    out = ensure_dir(Path("/content/mrc_tmp") / tag / f"seed{seed}")
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        weight_decay=0.01, warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="loss",
        greater_is_better=False, report_to="none", seed=seed, remove_unused_columns=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr_clean, eval_dataset=val_clean,
                      processing_class=tokenizer, data_collator=DefaultDataCollator(),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    trainer.train()
    write_json(out / "training_log.json", trainer.state.log_history)
    metrics = _squad_post_eval(model, tokenizer, ev_test, device)

    del trainer, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"f1": float(metrics["f1"]), "exact_match": float(metrics["exact_match"])}


def run_mrc_probe(arm_results, qa_splits, results_jsonl, *, task="MRC-ViQuAD",
                  seeds=(42, 123, 456), checkpoints=("final",), device="cuda",
                  force=False, **mrc_kwargs):
    """For each arm x checkpoint x seed: fine-tune MRC & append {f1, exact_match}.
    Skips (task, arm, ckpt, seed) already recorded. Writes to the shared hard-task file."""
    results_jsonl = Path(results_jsonl)
    done = read_jsonl(results_jsonl) if results_jsonl.exists() else []

    def is_done(arm, ckpt, seed):
        return any(r.get("task") == task and r.get("arm") == arm
                   and r.get("checkpoint") == ckpt and r.get("seed") == seed for r in done)

    for ar in arm_results:
        arm = ar["arm"]
        dirs = {"final": ar.get("final_dir"), "early": ar.get("early_dir")}
        for ckpt in checkpoints:
            md = dirs.get(ckpt)
            if not md or not Path(md).joinpath("model.safetensors").exists():
                print(f"  skip {task}/{arm}/{ckpt}: no model dir")
                continue
            for seed in seeds:
                if is_done(arm, ckpt, seed) and not force:
                    print(f"  skip existing {task}/{arm}/{ckpt}/seed{seed}")
                    continue
                print(f"  {task}: {arm}/{ckpt}/seed{seed} ...")
                m = finetune_mrc(md, qa_splits["train"], qa_splits["validation"],
                                 qa_splits["test"], seed=seed, device=device, **mrc_kwargs)
                if m is None:
                    continue
                row = {"task": task, "arm": arm, "checkpoint": ckpt, "seed": seed, **m}
                append_jsonl(results_jsonl, row)
                done.append(row)
                print(f"    -> {row}")
    return read_jsonl(results_jsonl)
