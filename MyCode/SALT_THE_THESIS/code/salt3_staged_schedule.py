"""WSD (Warmup-Stable-Decay) learning-rate machinery for staged continued-pretraining,
keyed on the GLOBAL optimizer step so a resumed/continued session NEVER re-warms.

Why a control-callback instead of an external LambdaLR: HF stepping an externally passed
scheduler is version-sensitive (and unverifiable on the Colab-only training box). A
TrainerCallback that overwrites ``optimizer.param_groups[*]['lr']` in ``on_step_begin`` —
BEFORE the optimizer step — makes the applied LR exactly ``wsd_lr_at(global_step)`` on every
transformers version. The optimizer stays external (AdamW) so its moments carry across
sessions; ``lr_scheduler_type="constant"`` is an inert placeholder the callback overrides.

Pure helpers (``wsd_lr_at``/``cooldown_lr_at``/``compute_warmup_steps``) are torch-free so the
CPU smoke test can verify the LR shape without importing torch.
"""
from __future__ import annotations

import math
from pathlib import Path

PEAK_LR_DEFAULT = 1e-4


# ── Pure LR shape (no torch) ─────────────────────────────────────────────────
def wsd_lr_at(global_step: int, *, peak_lr: float = PEAK_LR_DEFAULT, warmup_steps: int) -> float:
    """Warmup-then-flat: linear 0->peak over ``warmup_steps`` GLOBAL steps, then exactly
    ``peak_lr`` forever. Keyed on the global step, so once ``global_step >= warmup_steps``
    a fresh session starts flat at peak with no re-warm spike. Cooldown is a separate branch."""
    if warmup_steps <= 0:
        return peak_lr
    if global_step < warmup_steps:
        return peak_lr * global_step / warmup_steps
    return peak_lr


def cooldown_lr_at(local_step: int, *, peak_lr: float, cooldown_steps: int) -> float:
    """Cosine anneal peak->0 over ``cooldown_steps`` (the milestone cooldown branch).
    local_step 0 -> peak; local_step == cooldown_steps -> 0."""
    if cooldown_steps <= 0:
        return 0.0
    frac = min(1.0, local_step / cooldown_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * frac))


def onesqrt_lr_at(local_step: int, *, peak_lr: float, cooldown_steps: int) -> float:
    """1-sqrt anneal peak->0 over ``cooldown_steps``. local_step 0 -> peak; == cooldown_steps -> 0.
    Drops faster early than cosine then flattens near zero; the strongest cooldown shape for
    downstream quality at a fixed decay budget over fresh data (Hägele et al. 2024, 2405.18392)."""
    if cooldown_steps <= 0:
        return 0.0
    frac = min(1.0, local_step / cooldown_steps)
    return peak_lr * (1.0 - math.sqrt(frac))


# Cooldown anneal shapes, keyed by name. Pure (torch-free) so the CPU smoke test can verify
# the curve without importing torch.
COOLDOWN_SHAPES = {"cosine": cooldown_lr_at, "1-sqrt": onesqrt_lr_at}


def compute_warmup_steps(planned_ceiling_steps: int) -> int:
    """Warmup = 2% of the planned ceiling, clamped to [20, 500]. Scales from a tiny
    dry-run to a 10M-doc ceiling while always completing within the horizon."""
    return int(min(500, max(20, round(0.02 * planned_ceiling_steps))))


def cooldown_steps_for(budget_steps_so_far: int, *, cap: int = 1000, frac: float = 0.10) -> int:
    """~10% of budget-so-far, capped (bounded cooldown compute even at 10M)."""
    return int(min(cap, max(1, math.ceil(frac * budget_steps_so_far))))


# ── Optimizer carry (torch) ──────────────────────────────────────────────────
def build_wsd_optimizer(model, *, peak_lr: float = PEAK_LR_DEFAULT, weight_decay: float = 0.01):
    """AdamW only (no scheduler); the WSD LR is driven by the callback. Initial lr = peak so
    the inert constant placeholder, if it ever wins a step, lands on peak not on a stale value."""
    from torch.optim import AdamW

    # betas=(0.9, 0.95): the NeoBERT/LLaMA-2 AdamW recipe (NeoBERT conf/optimizer/adamw.yaml).
    # The lower beta2 tracks the second moment more responsively — the regime the body was
    # pretrained under, so CPT does not fight a differently-shaped preconditioner.
    return AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=weight_decay)


def save_optimizer_state(optimizer, path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), str(path))


def load_optimizer_state(optimizer, path) -> bool:
    """Restore Adam moments if a prior session saved them. Returns True on load."""
    import torch

    path = Path(path)
    if not path.exists():
        return False
    optimizer.load_state_dict(torch.load(str(path), map_location="cpu"))
    return True


# ── LR-control callbacks (torch/transformers, lazy) ──────────────────────────
def _set_all_lr(kwargs, lr: float) -> None:
    opt = kwargs.get("optimizer")
    if opt is not None:
        for group in opt.param_groups:
            group["lr"] = lr


def make_wsd_lr_callback(peak_lr: float, warmup_steps: int, global_step_offset: int = 0):
    """Set LR = ``wsd_lr_at(offset + local_step)`` before every step. Resume passes
    ``global_step_offset = manifest.global_step``; if that is >= warmup_steps the LR is flat
    at peak from local step 0 — the no-re-warm guarantee."""
    from transformers import TrainerCallback

    class _WSDLR(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            _set_all_lr(kwargs, wsd_lr_at(global_step_offset + state.global_step,
                                          peak_lr=peak_lr, warmup_steps=warmup_steps))
            return control

    return _WSDLR()


def make_cooldown_lr_callback(peak_lr: float, cooldown_steps: int, shape: str = "cosine",
                              warmup_steps: int = 0):
    """LR peak->0 over the cooldown branch (keyed on the branch-local step).

    ``shape`` picks the anneal curve: "cosine" (default, back-compat) or "1-sqrt".
    ``warmup_steps`` > 0 prepends a short linear 0->peak ramp before the anneal — used when the
    decay starts from a FRESH optimizer (no carried Adam moments) so the cold second-moment
    estimates re-stabilise before the LR bites; the anneal then runs over the remaining steps.
    With carried optimizer state, leave warmup_steps=0 for a pure anneal."""
    from transformers import TrainerCallback

    lr_fn = COOLDOWN_SHAPES.get(shape)
    if lr_fn is None:
        raise ValueError(f"unknown cooldown shape {shape!r}; expected one of {sorted(COOLDOWN_SHAPES)}")
    decay_steps = max(1, cooldown_steps - warmup_steps)

    class _CooldownLR(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            s = state.global_step
            if warmup_steps > 0 and s < warmup_steps:
                lr = peak_lr * (s + 1) / warmup_steps
            else:
                lr = lr_fn(s - warmup_steps, peak_lr=peak_lr, cooldown_steps=decay_steps)
            _set_all_lr(kwargs, lr)
            return control

    return _CooldownLR()


def _mask_all_collator(tokenizer, mlm_probability: float):
    """NeoBERT ``mask_all`` MLM collator built DIRECTLY (no override of any transformers internal),
    so it is correct on any transformers version and cannot silently fall back to BERT 80/10/10.

    Behaviour (identical to salt3_common.make_mlm_collator('mask_all')): pick ~``mlm_probability`` of
    non-special positions with a Bernoulli draw; every picked position -> [MASK] (100%, no random/
    keep); labels = the original id at picked positions and -100 everywhere else. Pre-packed chunks
    are fixed length, but we still right-pad a ragged batch to a multiple of 8 and never mask pads."""
    import torch

    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    if mask_id is None or (isinstance(mask_id, int) and mask_id < 0):
        raise ValueError("tokenizer has no [MASK] token; cannot do mask_all MLM")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(examples):
        rows = [list(e["input_ids"]) for e in examples]
        maxlen = max(len(r) for r in rows)
        maxlen += (-maxlen) % 8  # pad to a multiple of 8 (matches the reference pad_to_multiple_of=8)
        input_ids = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), maxlen), dtype=torch.long)
        special = torch.zeros((len(rows), maxlen), dtype=torch.bool)
        for i, r in enumerate(rows):
            n = len(r)
            input_ids[i, :n] = torch.tensor(r, dtype=torch.long)
            attention_mask[i, :n] = 1
            special[i, :n] = torch.tensor(
                tokenizer.get_special_tokens_mask(r, already_has_special_tokens=True), dtype=torch.bool)
        special |= attention_mask == 0  # never mask padding
        labels = input_ids.clone()
        prob = torch.full(input_ids.shape, mlm_probability)
        prob.masked_fill_(special, 0.0)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100
        input_ids[masked] = mask_id
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def assert_mask_all(collator, tokenizer, *, seq_len: int = 1024, batch: int = 8,
                    mlm_probability: float = 0.20) -> dict:
    """Run ``collator`` on a synthetic batch and HARD-FAIL unless it is true mask_all MLM. Cheap
    proof to run on the training box BEFORE a costly session: it catches a wrong/silently-degraded
    collator (e.g. BERT 80/10/10) immediately instead of after hours of training."""
    import torch

    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    lo = max((tokenizer.pad_token_id or 0) + 1, 5)  # benign content ids, away from special-token ids
    ex = [{"input_ids": [lo + ((i * seq_len + j) % 997) for j in range(seq_len)]} for i in range(batch)]
    out = collator([dict(e) for e in ex])
    ids, labels = out["input_ids"], out["labels"]
    orig = torch.tensor([e["input_ids"] for e in ex])
    masked = labels != -100
    assert masked.any(), "collator masked nothing"
    assert bool((ids[masked] == mask_id).all()), \
        "NOT mask_all: some masked positions are not [MASK] (random/keep present) -> wrong collator"
    assert bool((labels[masked] == orig[masked]).all()), "labels at masked positions != original tokens"
    assert bool((labels[~masked] == -100).all()), "labels off the masked set are not -100"
    frac = masked.float().mean().item()
    assert 0.5 * mlm_probability <= frac <= 1.5 * mlm_probability, \
        f"masked fraction {frac:.3f} far from target {mlm_probability}"
    return {"ok": True, "masked_fraction": round(frac, 4), "mask_token_id": int(mask_id)}


def build_mlm_collator_compat(make_mlm_collator, tokenizer, mlm_probability: float, mask_scheme: str):
    """Use salt3_common's collator when it supports ``mask_scheme``; otherwise fall back to a
    local builder so an older salt3_common (no mask_scheme arg) can't block the run NOR silently
    swap in BERT 80/10/10. mask_all -> local NeoBERT collator; bert_80_10_10 -> the stock collator.
    Shared by the staged trunk and the cooldown/decay branch so neither hard-depends on the
    salt3_common version present on the training box."""
    import inspect as _inspect

    if "mask_scheme" in _inspect.signature(make_mlm_collator).parameters:
        return make_mlm_collator(tokenizer, mlm_probability=mlm_probability, mask_scheme=mask_scheme)
    if mask_scheme == "mask_all":
        return _mask_all_collator(tokenizer, mlm_probability)
    return make_mlm_collator(tokenizer, mlm_probability=mlm_probability)  # legacy default == bert 80/10/10


# ── Milestone cooldown branch (forks a finished model; trunk untouched) ───────
def run_cooldown_branch(base_model_dir, *, cooldown_steps: int, peak_lr: float,
                        pool_train_ds, eval_ds, tokenizer, out_dir, device: str = "cuda",
                        mlm_probability: float = 0.20, mask_scheme: str = "mask_all",
                        weight_decay: float = 0.01, shape: str = "cosine",
                        opt_state_path=None, warmup_steps: int = 0,
                        snapshot_extra: dict | None = None) -> dict:
    """Load a FRESH copy of the trunk weights, anneal LR ``shape``->0 over ``pool_train_ds`` for
    ``cooldown_steps``, and save a standalone reloadable (NaN-free) model. The trunk model on disk
    is never touched.

    ``shape`` selects the anneal curve ("cosine" | "1-sqrt"). ``opt_state_path``, when given,
    restores the trunk's Adam moments so the anneal continues the same optimizer rather than a
    cold reset (preferred for downstream quality); ``warmup_steps`` re-warms a fresh optimizer
    when no carried state is available. ``pool_train_ds`` may be a recycled reserved pool OR a
    fresh contiguous unseen window — this function does not assume which."""
    import gc

    import torch
    from transformers import AutoModelForMaskedLM

    from salt3_common import (copy_neobert_remote_files, ensure_dir, eval_with_perplexity,
                              load_model_safe, make_mlm_collator, make_neo_mlm_trainer_class,
                              training_args, write_json)

    out_dir = ensure_dir(out_dir)
    model = load_model_safe(base_model_dir, model_cls=AutoModelForMaskedLM, device=device)
    optimizer = build_wsd_optimizer(model, peak_lr=peak_lr, weight_decay=weight_decay)
    carried = load_optimizer_state(optimizer, opt_state_path) if opt_state_path else False
    collator = build_mlm_collator_compat(make_mlm_collator, tokenizer, mlm_probability, mask_scheme)
    NeoMLMTrainer = make_neo_mlm_trainer_class()
    args = training_args(out_dir / "cooldown_ckpt", max_steps=cooldown_steps, warmup_ratio=0.0,
                         lr_scheduler_type="constant", eval_strategy="no", save_strategy="no",
                         load_best_model_at_end=False, save_total_limit=1)
    # carried moments -> no re-warm; cold reset -> brief warmup so the anneal does not bite cold
    eff_warmup = 0 if carried else warmup_steps
    trainer = NeoMLMTrainer(model=model, args=args, train_dataset=pool_train_ds,
                            eval_dataset=eval_ds, data_collator=collator,
                            processing_class=tokenizer, optimizers=(optimizer, None),
                            callbacks=[make_cooldown_lr_callback(peak_lr, cooldown_steps,
                                                                 shape=shape, warmup_steps=eff_warmup)])
    trainer.train()
    post = eval_with_perplexity(trainer)
    final_lr = optimizer.param_groups[0]["lr"]
    # per-step train-loss points (for a stitched flat-trunk -> decay progress plot); captured
    # before the trainer is freed
    loss_history = [{"step": h["step"], "loss": h["loss"]}
                    for h in trainer.state.log_history if "loss" in h]

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    copy_neobert_remote_files(out_dir)  # patched rotary/SwiGLU -> NaN-free reload
    info = {"kind": "cooled", "cooldown_steps": cooldown_steps, "peak_lr": peak_lr,
            "shape": shape, "optimizer": "carried" if carried else "fresh",
            "rewarmup_steps": eff_warmup, "final_lr": final_lr, "post_eval": post}
    info.update(snapshot_extra or {})
    write_json(out_dir / "snapshot_info.json", info)
    print(f"  cooldown branch -> {out_dir} (final_lr={final_lr:.2e}, eval_loss={post.get('eval_loss')})")

    del trainer, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"out_dir": str(out_dir), "final_lr": final_lr, "post_eval": post,
            "loss_history": loss_history, "optimizer": info["optimizer"]}
