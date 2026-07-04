"""Multi-arm CPT bake-off: train every init arm under an IDENTICAL budget and
overlay their training curves. Pairs with 15_bakeoff_cpt_compare_all_inits.ipynb.

Matched-budget protocol (why the comparison is fair):
  * same data slice, seed and hyperparameters for every arm
  * NO early stopping and NO load_best_model_at_end, so all arms stop at the same
    optimizer step and their curves are directly step-comparable
  * a stable "early" snapshot is saved once at ~early_frac of training so the
    downstream probe can measure the init advantage at a smaller data budget
Reuses salt3_common for the actual training machinery (DRY).
"""
from __future__ import annotations

import gc
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainerCallback

from salt3_common import (
    JsonlMetricsCallback, copy_neobert_remote_files, ensure_dir,
    eval_with_perplexity, load_model_safe, make_mlm_collator, make_mlm_datasets,
    make_neo_mlm_trainer_class, read_jsonl, train_with_run_resume, training_args,
    write_json,
)


def discover_arms(init_root, pattern: str = "trung_*"):
    """Return [(name, model_dir)] for every init dir matching pattern that holds a
    real model.safetensors. Sorted so arms compare in a stable order."""
    init_root = Path(init_root)
    arms = []
    for d in sorted(init_root.glob(pattern)):
        st = d / "model" / "model.safetensors"
        if st.exists():
            arms.append((d.name, d / "model"))
    return arms


class EarlySnapshotCallback(TrainerCallback):
    """Save a full model snapshot ONCE at the first step >= early_frac * max_steps.

    Independent of checkpoint rotation (save_total_limit), so the ~25%-budget
    artifact is guaranteed to survive for the downstream probe. Idempotent across
    a resumed run (skips if the snapshot already exists)."""

    def __init__(self, early_dir, tokenizer, early_frac: float = 0.25):
        self.early_dir = Path(early_dir)
        self.tokenizer = tokenizer
        self.early_frac = early_frac
        self._done = (self.early_dir / "model.safetensors").exists()

    def on_step_end(self, args, state, control, **kwargs):
        if self._done or not state.max_steps:
            return
        if state.global_step >= max(1, int(self.early_frac * state.max_steps)):
            model = kwargs["model"]
            ensure_dir(self.early_dir)
            model.save_pretrained(self.early_dir)
            self.tokenizer.save_pretrained(self.early_dir)
            copy_neobert_remote_files(self.early_dir)  # patched rotary/SwiGLU -> NaN-free reload
            write_json(self.early_dir / "snapshot_info.json", {
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "early_frac": self.early_frac,
            })
            print(f"  early snapshot @ step {state.global_step}/{state.max_steps} -> {self.early_dir}")
            self._done = True


def run_cpt_arm(arm_name, base_model_dir, runs_root, dataset_cache, *,
                data_cfg, train_cfg, early_frac: float = 0.25, save_early: bool = True,
                device: str = "cuda", force: bool = False):
    """Continue-pretrain one arm under the matched budget. Skips if final_model
    already exists (resume-friendly across Colab disconnects). Returns a dict with
    arm/metrics_path/final_dir/early_dir/status."""
    run_dir = ensure_dir(Path(runs_root) / arm_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    final_dir = run_dir / "final_model"
    early_dir = run_dir / "early_25pct"
    metrics_path = run_dir / "metrics.jsonl"

    has_final = final_dir.joinpath("model.safetensors").exists()
    if has_final and not force:
        print(f"[{arm_name}] final_model exists -> skip")
        return dict(arm=arm_name, metrics_path=metrics_path, final_dir=final_dir,
                    early_dir=early_dir if early_dir.joinpath("model.safetensors").exists() else None,
                    status="skipped")

    # force = a genuinely fresh run: drop checkpoints/snapshots so resume can't
    # pick up a stale trajectory and the early snapshot is recaptured.
    if force:
        for d in (ckpt_dir, final_dir, early_dir):
            shutil.rmtree(d, ignore_errors=True)
        ensure_dir(ckpt_dir)
    mode = "new" if force else "resume"

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    train_ds, eval_ds = make_mlm_datasets(
        tokenizer=tokenizer, cache_dir=dataset_cache,
        num_examples=data_cfg["num_examples"], max_seq_len=data_cfg["max_seq_len"],
        num_chunks=data_cfg.get("num_chunks"), eval_ratio=data_cfg["eval_ratio"],
        seed=data_cfg["seed"])
    collator = make_mlm_collator(tokenizer, mlm_probability=data_cfg["mlm_probability"])

    model = load_model_safe(base_model_dir, model_cls=AutoModelForMaskedLM, device=device)
    NeoMLMTrainer = make_neo_mlm_trainer_class()
    # matched budget: keep dense eval + rotating checkpoints, but force OFF anything
    # that would let arms diverge in stop-step or restore a non-final-budget model.
    overrides = {k: v for k, v in train_cfg.items() if k != "load_best_model_at_end"}
    args = training_args(ckpt_dir, load_best_model_at_end=False,
                         run_name=f"bakeoff100k-{arm_name}", **overrides)

    metrics_cb = JsonlMetricsCallback(metrics_path)
    callbacks = [metrics_cb.callback]
    if save_early:
        callbacks.append(EarlySnapshotCallback(early_dir, tokenizer, early_frac))

    trainer = NeoMLMTrainer(model=model, args=args, train_dataset=train_ds,
                            eval_dataset=eval_ds, data_collator=collator,
                            processing_class=tokenizer, callbacks=callbacks)

    pre = eval_with_perplexity(trainer)
    print(f"[{arm_name}] pre-CPT eval: {pre}")
    # 'resume' picks up a checkpoint if one exists, else starts from the base model;
    # 'new' (force) ignores any checkpoint and trains from the base model
    result = train_with_run_resume(trainer, mode)
    post = eval_with_perplexity(trainer)
    print(f"[{arm_name}] post-CPT eval: {post} (steps={result.global_step})")

    ensure_dir(final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    copy_neobert_remote_files(final_dir)
    write_json(run_dir / "cpt_summary.json",
               {"arm": arm_name, "pre": pre, "post": post, "global_step": result.global_step})

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(arm=arm_name, metrics_path=metrics_path, final_dir=final_dir,
                early_dir=early_dir if save_early else None, status="trained")


def overlay_comparison_curves(arm_metrics: dict, save_dir):
    """arm_metrics: {arm_name: metrics_path}. Saves 3 step-matched overlays
    (train loss, eval loss, eval perplexity vs optimizer step) and returns paths."""
    import math
    import matplotlib.pyplot as plt

    save_dir = ensure_dir(save_dir)

    def series(rows, key):
        by_step: dict[int, float] = {}
        for r in rows:
            if key in r and "step" in r:
                by_step[r["step"]] = r[key]  # last write per step wins (resume-safe)
        return sorted(by_step.items())

    panels = [
        ("loss", "Train loss", "train_loss_overlay.png", False),
        ("eval_loss", "Eval loss", "eval_loss_overlay.png", False),
        ("eval_loss", "Eval perplexity", "eval_ppl_overlay.png", True),
    ]
    paths = []
    loaded = {arm: read_jsonl(mp) for arm, mp in arm_metrics.items()}
    for key, title, fname, as_ppl in panels:
        plt.figure(figsize=(9, 6))
        plotted = False
        for arm, rows in loaded.items():
            pts = series(rows, key)
            if not pts:
                continue
            xs = [s for s, _ in pts]
            ys = [math.exp(min(v, 20)) if as_ppl else v for _, v in pts]
            plt.plot(xs, ys, marker="o", ms=3, label=arm)
            plotted = True
        plt.xlabel("Optimizer step (≈ data budget)")
        plt.ylabel(title)
        plt.title(f"{title} vs budget — all arms")
        if as_ppl:
            plt.yscale("log")
        plt.grid(alpha=0.3)
        if plotted:
            plt.legend(fontsize=8)
        plt.tight_layout()
        p = save_dir / fname
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.show()
        paths.append(p)
    return paths
