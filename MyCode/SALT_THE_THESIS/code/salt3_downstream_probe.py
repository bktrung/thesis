"""Light downstream probe for the CPT bake-off: fine-tune each arm's CPT'd model
on UIT-VSFC sentiment with a FIXED recipe (no HP grid) and summarize the
sweet-spot picture (early vs final checkpoint, gap to the random control).

Deliberately slim vs notebook 03's full grid+significance runner: the goal here
is a cheap "does the init still help downstream after a little training" signal
across many arms, not a publication-grade per-task protocol.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, Trainer, TrainingArguments,
)

from salt3_common import (
    NeoBERTSequenceClassifier, append_jsonl, ensure_dir, load_model_safe,
    read_jsonl, set_seed,
)


def compute_cls_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def load_vsfc():
    """UIT-VSFC -> DatasetDict with 'text'/'labels' (3-class sentiment)."""
    from datasets import Dataset, DatasetDict, load_dataset

    raw = load_dataset("tridm/UIT-VSFC")
    if "validation" not in raw and "dev" in raw:
        raw["validation"] = raw["dev"]
    cols = raw["train"].column_names
    tcol = next(c for c in ["Sentence", "sentence", "text"] if c in cols)
    lcol = next(c for c in ["Encoded_sentiment", "sentiment", "label"] if c in cols)

    def clean(s):
        return Dataset.from_dict({
            "text": [str(x) for x in s[tcol]],
            "labels": [int(x) for x in s[lcol]],
        })

    return DatasetDict({k: clean(raw[k]) for k in ["train", "validation", "test"]})


def load_vianli():
    """ViANLI (native Vietnamese NLI) -> DatasetDict with 'premise'/'hypothesis'/'labels'
    (3-class: entailment=0, neutral=1, contradiction=2). Builds val/test if absent."""
    from datasets import Dataset, DatasetDict, load_dataset

    raw = load_dataset("uitnlp/ViANLI")
    cols = raw[list(raw.keys())[0]].column_names
    pre = next(c for c in ["premise", "sentence1", "context"] if c in cols)
    hyp = next(c for c in ["hypothesis", "sentence2"] if c in cols)
    lab = next(c for c in ["label", "gold_label", "labels"] if c in cols)
    nli_map = {"entailment": 0, "neutral": 1, "contradiction": 2,
               "e": 0, "n": 1, "c": 2, "0": 0, "1": 1, "2": 2}

    def enc(v):
        return v if isinstance(v, int) else nli_map[str(v).strip().lower()]

    def clean(s):
        return Dataset.from_dict({
            "premise": [str(a) for a in s[pre]],
            "hypothesis": [str(b) for b in s[hyp]],
            "labels": [enc(x) for x in s[lab]],
        })

    keys = set(raw.keys())
    train = raw["train"]
    val = raw["validation"] if "validation" in keys else raw.get("dev")
    test = raw["test"] if "test" in keys else None
    if val is None:
        sp = train.train_test_split(test_size=0.1, seed=42); train, val = sp["train"], sp["test"]
    if test is None:
        sp = val.train_test_split(test_size=0.5, seed=42); val, test = sp["train"], sp["test"]
    return DatasetDict({"train": clean(train), "validation": clean(val), "test": clean(test)})


def finetune_classification(model_dir, dataset, num_labels: int = 3, *,
                            text_fields=("text",), train_cap=None,
                            metric_for_best: str = "f1_weighted", seed: int = 42,
                            lr: float = 2e-5, max_epochs: int = 5, batch_size: int = 16,
                            max_len: int = 256, patience: int = 2, device: str = "cuda"):
    """Fine-tune a NeoBERTSequenceClassifier on a cleaned DatasetDict; return test
    accuracy + f1_weighted + f1_macro. Best-on-validation model restored before testing.
    text_fields=("text",) for single-sentence; ("premise","hypothesis") for pair tasks."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base = load_model_safe(model_dir, model_cls=AutoModelForMaskedLM, device="cpu")
    model = NeoBERTSequenceClassifier(base, num_labels=num_labels).to(device)

    fields = tuple(text_fields)

    def tok(b):
        if len(fields) == 2:
            return tokenizer(b[fields[0]], b[fields[1]], truncation=True, max_length=max_len)
        return tokenizer(b[fields[0]], truncation=True, max_length=max_len)

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    splits = {}
    for k in ["train", "validation", "test"]:
        ds = dataset[k]
        if k == "train" and train_cap and len(ds) > train_cap:
            ds = ds.shuffle(seed=seed).select(range(train_cap))  # few-shot regime
        ds = ds.map(tok, batched=True)
        splits[k] = ds.remove_columns([c for c in ds.column_names if c not in keep])

    # key by arm AND checkpoint leaf (final_model vs early_25pct) so the two
    # checkpoints of one arm don't collide in the same scratch dir
    tag = f"{Path(model_dir).parent.name}__{Path(model_dir).name}"
    out = ensure_dir(Path("/content/ft_tmp") / tag / f"seed{seed}")
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model=metric_for_best,
        greater_is_better=True, save_total_limit=1, logging_steps=50,
        report_to="none", seed=seed,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=splits["train"],
        eval_dataset=splits["validation"], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_cls_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()
    test_metrics = trainer.evaluate(eval_dataset=splits["test"], metric_key_prefix="test")

    del trainer, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "accuracy": test_metrics.get("test_accuracy"),
        "f1_weighted": test_metrics.get("test_f1_weighted"),
        "f1_macro": test_metrics.get("test_f1_macro"),
    }


def run_downstream_probe(arm_results, dataset, results_jsonl, *, num_labels: int = 3,
                         seeds=(42, 123), checkpoints=("final", "early"),
                         device: str = "cuda", force: bool = False, **ft_kwargs):
    """For each arm x checkpoint x seed: fine-tune & append test metrics. Skips
    (arm, checkpoint, seed) already recorded so the loop is disconnect-resumable.
    arm_results: list of run_cpt_arm() dicts (arm, final_dir, early_dir)."""
    results_jsonl = Path(results_jsonl)
    done = read_jsonl(results_jsonl) if results_jsonl.exists() else []

    def is_done(arm, ckpt, seed):
        return any(r.get("arm") == arm and r.get("checkpoint") == ckpt and r.get("seed") == seed
                   for r in done)

    for ar in arm_results:
        arm = ar["arm"]
        dirs = {"final": ar.get("final_dir"), "early": ar.get("early_dir")}
        for ckpt in checkpoints:
            md = dirs.get(ckpt)
            if not md or not Path(md).joinpath("model.safetensors").exists():
                print(f"  skip {arm}/{ckpt}: no model dir")
                continue
            for seed in seeds:
                if is_done(arm, ckpt, seed) and not force:
                    print(f"  skip existing {arm}/{ckpt}/seed{seed}")
                    continue
                print(f"  fine-tune {arm}/{ckpt}/seed{seed} ...")
                m = finetune_classification(md, dataset, num_labels=num_labels,
                                            seed=seed, device=device, **ft_kwargs)
                row = {"arm": arm, "checkpoint": ckpt, "seed": seed, **m}
                append_jsonl(results_jsonl, row)
                done.append(row)
                print(f"    -> {row}")
    return read_jsonl(results_jsonl)


def summarize_sweet_spot(results_jsonl, save_dir,
                         random_arms=("trung_random_meannorm", "trung_naive_random_zerobias")):
    """Aggregate results -> CSV + final-budget bar (with random baseline line) +
    early-vs-final delta plot (does the init advantage shrink with more data?)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    save_dir = ensure_dir(save_dir)
    df = pd.DataFrame(read_jsonl(results_jsonl))
    if df.empty:
        print("no downstream results yet")
        return df

    agg = (df.groupby(["arm", "checkpoint"])[["accuracy", "f1_weighted"]]
             .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    csv = save_dir / "downstream_summary.csv"
    agg.to_csv(csv, index=False)
    print("saved", csv)

    # final-budget F1 per arm, with the best random control as a reference line
    fin = agg[agg["checkpoint"] == "final"].sort_values("f1_weighted_mean", ascending=False)
    if not fin.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(fin["arm"], fin["f1_weighted_mean"],
                yerr=fin["f1_weighted_std"].fillna(0), capsize=3)
        rnd = fin[fin["arm"].isin(random_arms)]["f1_weighted_mean"]
        if len(rnd):
            plt.axhline(rnd.max(), ls="--", c="r", label="best random control")
            plt.legend()
        plt.ylabel("VSFC F1-weighted (final @100k)")
        plt.title("Downstream performance per arm")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "downstream_final_bar.png", dpi=150, bbox_inches="tight")
        plt.show()

    # early vs final: where does the init advantage live?
    piv = df.groupby(["arm", "checkpoint"])["f1_weighted"].mean().unstack("checkpoint")
    if {"early", "final"}.issubset(piv.columns):
        piv = piv.sort_values("final", ascending=False)
        x = list(range(len(piv)))
        plt.figure(figsize=(10, 5))
        plt.plot(x, piv["early"], "o-", label="early (~25% budget)")
        plt.plot(x, piv["final"], "s-", label="final (100% budget)")
        plt.xticks(x, piv.index, rotation=45, ha="right")
        plt.ylabel("VSFC F1-weighted")
        plt.title("Early vs final — where does the init advantage live?")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "downstream_early_vs_final.png", dpi=150, bbox_inches="tight")
        plt.show()
    return agg


def run_seq_probe(arm_results, results_jsonl, *, task: str, dataset, num_labels: int,
                  text_fields=("text",), metric_for_best: str = "f1_weighted",
                  seeds=(42, 123, 456), checkpoints=("final",), train_cap=None,
                  device: str = "cuda", force: bool = False, **ft_kwargs):
    """Task-keyed sequence-task probe over saved checkpoints. Rows carry a 'task'
    field so several tasks share one results file; skips (task, arm, ckpt, seed)
    already recorded. Works for single-text (VSFC) and pair tasks (ViANLI)."""
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
                m = finetune_classification(md, dataset, num_labels=num_labels,
                                            text_fields=text_fields, train_cap=train_cap,
                                            metric_for_best=metric_for_best, seed=seed,
                                            device=device, **ft_kwargs)
                row = {"task": task, "arm": arm, "checkpoint": ckpt, "seed": seed, **m}
                append_jsonl(results_jsonl, row)
                done.append(row)
                print(f"    -> {row}")
    return read_jsonl(results_jsonl)


def summarize_tasks(results_jsonl, save_dir, task_metric: dict,
                    random_arms=("trung_random_meannorm", "trung_naive_random_zerobias")):
    """Per-task leaderboard over the 'final' checkpoint. task_metric maps
    {task_name: metric_column}. Writes one CSV + a ranked bar per task and prints
    a leaderboard (best arm first, gap to best random control)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    save_dir = ensure_dir(save_dir)
    df = pd.DataFrame(read_jsonl(results_jsonl))
    if df.empty:
        print("no hard-task results yet")
        return df
    df = df[df["checkpoint"] == "final"]
    rows = []
    for task, metric in task_metric.items():
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        if metric not in sub.columns:
            # older rows may predate a metric (e.g. f1_macro added later) — skip, don't crash
            print(f"  skip {task}: metric {metric!r} not in results (have {[c for c in sub.columns if c not in ('task','arm','checkpoint','seed')]})")
            continue
        g = sub.groupby("arm")[metric].agg(["mean", "std", "count"]).reset_index()
        g = g.sort_values("mean", ascending=False)
        g["task"], g["metric"] = task, metric
        rows.append(g)
        best_rnd = g[g["arm"].isin(random_arms)]["mean"].max()
        print(f"\n== {task} ({metric}) — best random control = {best_rnd:.4f} ==")
        for _, r in g.iterrows():
            tag = "  <- random" if r["arm"] in random_arms else ""
            print(f"  {r['arm']:32s} {r['mean']:.4f} ± {r['std'] if pd.notna(r['std']) else 0:.4f}"
                  f"  (Δrand {r['mean']-best_rnd:+.4f}, n={int(r['count'])}){tag}")
        plt.figure(figsize=(10, 5))
        plt.bar(g["arm"], g["mean"], yerr=g["std"].fillna(0), capsize=3)
        plt.axhline(best_rnd, ls="--", c="r", label="best random control")
        plt.ylabel(f"{task} {metric} (final)")
        plt.title(f"{task}: init ranking @100k CPT")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"hardtask_{task.replace('/', '_')}_bar.png", dpi=150, bbox_inches="tight")
        plt.show()
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        csv = save_dir / "hardtask_summary.csv"
        out.to_csv(csv, index=False)
        print("\nsaved", csv)
    return out
