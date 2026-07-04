"""Fresh-data WSD decay branch: fork a parked trunk snapshot and anneal LR -> 0 over a
CONTIGUOUS, previously-UNSEEN data window (no recycled pool, no repeat).

This is the "proper" cooldown that the staged harness's bounded recycled-pool cooldown only
approximates: the decay phase — where downstream quality consolidates — finally sees fresh
tokens instead of a tiny held-out pool looped many times.

Manifest-neutral by design: reads the trunk snapshot + manifest position READ-ONLY and writes a
standalone ``milestone_<docs>_decay`` dir. It never advances/marks/saves the trunk manifest, so a
later staged-CPT session still continues the flat trunk from exactly where it stopped.

Usage (Colab, after the trunk is parked at a milestone):
    from salt3_fresh_decay import run_fresh_decay
    run_fresh_decay(arm_dir=..., base_snapshot=arm_dir/'milestone_4000000_stable',
                    tokenizer=tok, dataset_cache=..., ceiling_docs=..., chunks_consumed=...,
                    target_total_docs=5_000_000, decay_frac=0.20, shape='1-sqrt',
                    opt_state_path=arm_dir/'opt_state.pt')   # opt_state only if trunk parked at 4M
"""
from __future__ import annotations

from pathlib import Path

import salt3_staged_schedule as sched
from salt3_common import (chunks_to_steps, docs_to_chunks, ensure_dir, make_mlm_datasets,
                          read_cache_meta, write_json)


def plan_decay_window(*, chunks_consumed: int, train_chunks: int, ratio: float, eff_batch: int,
                      target_total_docs: int, decay_frac: float = 0.20) -> dict:
    """Pure planner (torch-free, CPU-testable). Given the trunk position and cache size, return
    the fresh decay window + whole-step count, clamped to the unseen tail. Reuse-first: aims for
    ``decay_frac`` of the target budget but shortens to whatever unseen data remains rather than
    rebuild or repeat — the achieved fraction is reported so the caller can label it honestly."""
    total_steps = chunks_to_steps(docs_to_chunks(target_total_docs, ratio), eff_batch)
    want_steps = round(decay_frac * total_steps)
    unseen = max(0, train_chunks - chunks_consumed)
    decay_steps = min(want_steps, unseen // eff_batch)   # floor -> whole steps, single pass, no repeat
    decay_chunks = decay_steps * eff_batch
    return {
        "chunk_start": int(chunks_consumed),
        "chunk_end": int(chunks_consumed + decay_chunks),
        "decay_chunks": int(decay_chunks),
        "decay_steps": int(decay_steps),
        "unseen_chunks": int(unseen),
        "total_steps": int(total_steps),
        "achieved_frac": round(decay_steps / total_steps, 4) if total_steps else 0.0,
    }


def run_fresh_decay(*, arm_dir, base_snapshot, tokenizer, dataset_cache, ceiling_docs: int,
                    chunks_consumed: int, max_seq_len: int = 1024, eval_ratio: float = 0.02,
                    seed: int = 42, per_device_bs: int = 32, grad_accum: int = 16,
                    target_total_docs: int = 5_000_000, decay_frac: float = 0.20,
                    peak_lr: float = 1e-4, shape: str = "1-sqrt", opt_state_path=None,
                    rewarmup_steps: int = 300, eval_chunks: int = 1024,
                    mlm_probability: float = 0.20, mask_scheme: str = "mask_all",
                    device: str = "cuda") -> dict:
    """Anneal a parked trunk snapshot over a fresh unseen window. opt_state_path carries the
    trunk's Adam moments (preferred, only valid if the trunk is parked at the snapshot's budget);
    otherwise a fresh optimizer re-warms over ``rewarmup_steps``."""
    arm_dir, base = Path(arm_dir), Path(base_snapshot)
    eff_batch = per_device_bs * grad_accum

    meta = read_cache_meta(dataset_cache, tokenizer, ceiling_docs, max_seq_len, eval_ratio, seed)
    if meta is None:
        raise FileNotFoundError(f"no cache_meta for ceiling {ceiling_docs:,}; build the ceiling cache first")
    ratio, train_chunks = meta["ratio_chunks_per_doc"], meta["train_chunks"]

    win = plan_decay_window(chunks_consumed=chunks_consumed, train_chunks=train_chunks, ratio=ratio,
                            eff_batch=eff_batch, target_total_docs=target_total_docs, decay_frac=decay_frac)
    if win["decay_steps"] < 1:
        raise ValueError(f"no unseen data for a fresh decay (unseen={win['unseen_chunks']} chunks "
                         f"< one step of {eff_batch}); extend the ceiling cache")
    if not (base / "model.safetensors").exists():
        raise FileNotFoundError(f"base snapshot weights missing: {base}")

    # fresh, contiguous, previously-unseen window starting exactly where the trunk stopped
    train_ds, eval_ds = make_mlm_datasets(
        tokenizer=tokenizer, cache_dir=dataset_cache, num_examples=ceiling_docs,
        max_seq_len=max_seq_len, eval_ratio=eval_ratio, seed=seed,
        chunk_start=win["chunk_start"], chunk_end=win["chunk_end"], eval_chunks=eval_chunks)
    # the realized slice must match the planned fresh window exactly: catches a silent clamp/shift
    # in the dataset slicer (e.g. the window running past the cache tail) that would otherwise feed
    # short or misaligned data. Since the planner sets decay_chunks == decay_steps * eff_batch, this
    # also guarantees no-repeat (a full single pass at the requested step count).
    assert len(train_ds) == win["decay_chunks"], (
        f"realized window {len(train_ds)} != planned {win['decay_chunks']} chunks "
        "-> cache slice clamped/shifted; decay aborted")

    out_dir = ensure_dir(arm_dir / f"milestone_{int(target_total_docs)}_decay")
    method = f"fresh_culturax_{shape}_{int(round(win['achieved_frac'] * 100))}pct"
    info = sched.run_cooldown_branch(
        base, cooldown_steps=win["decay_steps"], peak_lr=peak_lr, pool_train_ds=train_ds,
        eval_ds=eval_ds, tokenizer=tokenizer, out_dir=out_dir, device=device,
        mlm_probability=mlm_probability, mask_scheme=mask_scheme, shape=shape,
        opt_state_path=opt_state_path, warmup_steps=rewarmup_steps,
        snapshot_extra={"kind": "decay", "method": method, "docs": int(target_total_docs),
                        "decay_steps": win["decay_steps"], "decay_chunks": win["decay_chunks"],
                        "achieved_frac": win["achieved_frac"], "window": [win["chunk_start"], win["chunk_end"]]})
    write_json(out_dir / "decay_plan.json", win)
    print(f"[fresh-decay] {out_dir.name}: {win['decay_steps']:,} steps over fresh chunks "
          f"[{win['chunk_start']:,},{win['chunk_end']:,}) ~{win['achieved_frac']*100:.1f}% | "
          f"optimizer={info.get('optimizer')} shape={shape}")
    return {"out_dir": str(out_dir), "window": win, **info}
