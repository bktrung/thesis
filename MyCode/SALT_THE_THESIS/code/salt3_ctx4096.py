"""Context-extension (stage-2 analog): fork the parked ``milestone_5000000_decay`` snapshot and
continue at sequence length 4096 over a CONTIGUOUS, previously-UNSEEN data window, at a constant
low LR. NeoBERT pretrains in two stages (1024 then 50k steps @4096); our CPT stopped at the
stage-1 equivalent, so this is the missing stage-2 finetune.

Reuse-first, no re-streaming: the existing seq-1024 ceiling cache packs each split's token stream
sequentially (salt3_common.make_mlm_datasets), so FOUR consecutive same-split 1024-chunks form ONE
contiguous 4096 chunk. We slice the unseen 1024-tail past ``chunks_consumed`` and regroup it into
4096 rows — no new tokenization pass.

Manifest-neutral by design (mirrors salt3_fresh_decay): reads the base snapshot READ-ONLY and
writes a standalone ``milestone_5000000_ctx4096`` dir. It never advances/marks the trunk manifest.

LR is a warmup->constant plateau (``make_wsd_lr_callback``), NOT a cooldown: this stage extends the
context, it does not re-anneal. Fresh AdamW (β2=0.95) re-warms over ``warmup_steps`` then holds
``peak_lr`` (10% of the 1e-4 CPT peak).

Usage (Colab, after milestone_5000000_decay is parked on Drive):
    from salt3_ctx4096 import run_ctx4096_extend
    run_ctx4096_extend(arm_dir=..., base_snapshot=arm_dir/'milestone_5000000_decay',
                       tokenizer=tok, dataset_cache=..., ceiling_docs=5_000_000,
                       chunks_consumed=<decay chunk_end>, per_device_bs=8, grad_accum=16)
"""
from __future__ import annotations

from pathlib import Path

import salt3_staged_schedule as sched
from salt3_common import (ensure_dir, get_last_checkpoint, make_mlm_datasets, read_cache_meta,
                          write_json)

BASE_SEQ_LEN = 1024  # the seq length of the existing ceiling cache we regroup from


def plan_ctx4096_window(*, chunks_consumed: int, train_chunks_1024: int, group: int = 4,
                        eff_batch: int, target_tokens: int = 200_000_000) -> dict:
    """Pure planner (torch-free, CPU-testable). Size a single-pass 4096 run to ``target_tokens``,
    clamped to whatever unseen 1024-tail remains past ``chunks_consumed``. Reuse-first: shortens
    to the available data rather than repeat or rebuild, and reports the achieved token budget.

    Units: ``group`` seq-1024 chunks concatenate into one seq-4096 chunk (group*1024 tokens).
    ``eff_batch`` counts 4096-chunks/step, so one step = ``eff_batch*group*1024`` tokens.
    """
    tokens_per_step = eff_batch * group * BASE_SEQ_LEN
    want_steps = round(target_tokens / tokens_per_step)
    unseen_1024 = max(0, train_chunks_1024 - chunks_consumed)
    avail_4096 = unseen_1024 // group                     # whole 4096-chunks in the unseen tail
    steps = min(want_steps, avail_4096 // eff_batch)      # floor -> whole steps, single pass, no repeat
    n_4096_chunks = steps * eff_batch
    n_1024_chunks = n_4096_chunks * group                 # exact 1024-slice the regroup consumes
    achieved_tokens = n_1024_chunks * BASE_SEQ_LEN
    assert n_1024_chunks <= unseen_1024, "planned 1024-slice exceeds the unseen tail"
    return {
        "chunk_start": int(chunks_consumed),
        "chunk_end": int(chunks_consumed + n_1024_chunks),
        "n_4096_chunks": int(n_4096_chunks),
        "n_1024_chunks": int(n_1024_chunks),
        "steps": int(steps),
        "unseen_1024_chunks": int(unseen_1024),
        "seq_len": int(group * BASE_SEQ_LEN),
        "want_steps": int(want_steps),
        "target_tokens": int(target_tokens),
        "achieved_tokens": int(achieved_tokens),
        "achieved_frac": round(achieved_tokens / target_tokens, 4) if target_tokens else 0.0,
    }


def _regroup_rows(rows_iter, group: int):
    """Pure-python (torch/datasets-free) core of the regroup: yield the concatenation of every
    ``group`` consecutive input_ids lists; drop the ragged tail remainder. Because the source cache
    packs each split's token stream sequentially, concatenating 4 consecutive 1024-chunks
    reconstructs the exact contiguous 4096-token span — verified byte-for-byte in the CPU checks."""
    buf: list = []
    for row in rows_iter:
        buf.append(row)
        if len(buf) == group:
            merged: list = []
            for b in buf:
                merged.extend(b)
            yield merged
            buf = []


def regroup_to_4096(ds_1024_slice, group: int = 4):
    """Regroup a contiguous same-split seq-1024 HF Dataset slice into a seq-(group*1024) Dataset.
    Every output row is exactly ``group*1024`` long; the tail remainder (< group rows) is dropped."""
    from datasets import Dataset

    merged = list(_regroup_rows(ds_1024_slice["input_ids"], group))
    return Dataset.from_dict({"input_ids": merged})


def run_ctx4096_extend(*, arm_dir, base_snapshot, tokenizer, dataset_cache, ceiling_docs: int,
                       chunks_consumed: int, group: int = 4, eval_ratio: float = 0.02,
                       seed: int = 42, per_device_bs: int = 8, grad_accum: int = 16,
                       target_tokens: int = 200_000_000, peak_lr: float = 1e-5,
                       warmup_steps: int = 50, mlm_probability: float = 0.20,
                       mask_scheme: str = "mask_all", save_steps: int = 100,
                       eval_chunks: int = 1024, resume: bool = True, device: str = "cuda") -> dict:
    """Fork ``base_snapshot`` (the decay milestone) and continue at seq 4096 over the unseen tail
    at constant ``peak_lr`` for the planner-derived step count. Writes a resumable, manifest-neutral
    ``milestone_5000000_ctx4096`` dir; the trunk/decay checkpoints on disk are never touched."""
    import gc

    import torch
    from transformers import AutoModelForMaskedLM

    from salt3_common import (copy_neobert_remote_files, eval_with_perplexity, load_model_safe,
                              make_mlm_collator, make_neo_mlm_trainer_class, training_args)

    arm_dir, base = Path(arm_dir), Path(base_snapshot)
    eff_batch = per_device_bs * grad_accum
    # The single-pass/no-repeat guarantee sizes max_steps and the window with
    # eff_batch = per_device_bs*grad_accum and no world_size factor; under DDP the sharded
    # dataloader would wrap-and-repeat (or under-consume) the window, silently breaking it.
    # This stage is a single-A100 session by design — fail loudly if that ever changes.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        raise RuntimeError(f"ctx4096 assumes a single device; found {torch.cuda.device_count()} GPUs "
                           "-> the no-repeat window math needs a world_size factor before multi-GPU use")

    meta = read_cache_meta(dataset_cache, tokenizer, ceiling_docs, BASE_SEQ_LEN, eval_ratio, seed)
    if meta is None:
        raise FileNotFoundError(f"no cache_meta for ceiling {ceiling_docs:,}; build the ceiling cache first")
    train_chunks_1024 = meta["train_chunks"]

    win = plan_ctx4096_window(chunks_consumed=chunks_consumed, train_chunks_1024=train_chunks_1024,
                              group=group, eff_batch=eff_batch, target_tokens=target_tokens)
    if win["steps"] < 1:
        raise ValueError(f"no unseen data for a 4096 run (unseen={win['unseen_1024_chunks']} 1024-chunks "
                         f"< one step of {eff_batch*group}); extend the ceiling cache")
    if not (base / "model.safetensors").exists():
        raise FileNotFoundError(f"base snapshot weights missing: {base}")

    # contiguous unseen seq-1024 window, then regroup 4x into seq-4096 (train + a small eval set)
    train_1024, eval_1024 = make_mlm_datasets(
        tokenizer=tokenizer, cache_dir=dataset_cache, num_examples=ceiling_docs,
        max_seq_len=BASE_SEQ_LEN, eval_ratio=eval_ratio, seed=seed,
        chunk_start=win["chunk_start"], chunk_end=win["chunk_end"], eval_chunks=eval_chunks)
    # the realized 1024-slice must match the plan exactly: catches a silent cache clamp/shift that
    # would feed short/misaligned data; also guarantees the single-pass no-repeat window.
    assert len(train_1024) == win["n_1024_chunks"], (
        f"realized window {len(train_1024)} != planned {win['n_1024_chunks']} 1024-chunks "
        "-> cache slice clamped/shifted; ctx4096 aborted")
    train_ds = regroup_to_4096(train_1024, group=group)
    eval_ds = regroup_to_4096(eval_1024, group=group)
    assert len(train_ds) == win["n_4096_chunks"], (
        f"regrouped {len(train_ds)} != planned {win['n_4096_chunks']} 4096-chunks")

    out_dir = ensure_dir(arm_dir / "milestone_5000000_ctx4096")
    model = load_model_safe(base, model_cls=AutoModelForMaskedLM, device=device)
    optimizer = sched.build_wsd_optimizer(model, peak_lr=peak_lr)  # fresh AdamW β2=0.95
    collator = sched.build_mlm_collator_compat(make_mlm_collator, tokenizer, mlm_probability, mask_scheme)
    NeoMLMTrainer = make_neo_mlm_trainer_class()
    # save_strategy='steps' + save_total_limit=1 -> resumable; the WSD callback drives the constant
    # plateau, so lr_scheduler_type='constant'/warmup_ratio=0 are inert placeholders.
    args = training_args(out_dir / "ctx4096_ckpt", max_steps=win["steps"], warmup_ratio=0.0,
                         lr_scheduler_type="constant", eval_strategy="no", save_strategy="steps",
                         save_steps=save_steps, save_total_limit=1, load_best_model_at_end=False,
                         per_device_train_batch_size=per_device_bs, per_device_eval_batch_size=per_device_bs,
                         gradient_accumulation_steps=grad_accum)
    trainer = NeoMLMTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                            data_collator=collator, processing_class=tokenizer,
                            optimizers=(optimizer, None),
                            callbacks=[sched.make_wsd_lr_callback(peak_lr, warmup_steps)])
    ckpt = get_last_checkpoint(str(args.output_dir)) if resume else None
    if ckpt:
        print(f"[ctx4096] resuming from {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    final_lr = optimizer.param_groups[0]["lr"]
    loss_history = [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]

    # save the trained weights BEFORE post-eval, so a transient eval failure never loses a finished run
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    copy_neobert_remote_files(out_dir)  # patched rotary/SwiGLU -> NaN-free reload
    post = eval_with_perplexity(trainer)
    info = {"kind": "ctx4096", "seq_len": win["seq_len"], "steps": win["steps"], "peak_lr": peak_lr,
            "warmup_steps": warmup_steps, "final_lr": final_lr, "achieved_tokens": win["achieved_tokens"],
            "achieved_frac": win["achieved_frac"], "window": [win["chunk_start"], win["chunk_end"]],
            "post_eval": post}
    write_json(out_dir / "snapshot_info.json", info)
    write_json(out_dir / "ctx4096_plan.json", win)
    print(f"[ctx4096] {out_dir.name}: {win['steps']:,} steps @seq{win['seq_len']} over unseen 1024-chunks "
          f"[{win['chunk_start']:,},{win['chunk_end']:,}) ~{win['achieved_tokens']/1e6:.0f}M tok "
          f"({win['achieved_frac']*100:.0f}% of target) | final_lr={final_lr:.2e} eval_loss={post.get('eval_loss')}")

    del trainer, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"out_dir": str(out_dir), "window": win, "final_lr": final_lr,
            "post_eval": post, "loss_history": loss_history}
