"""Staged continued-pretraining session runner: ties data windowing (salt3_common),
the WSD schedule (salt3_staged_schedule), the manifest (salt3_staged_cpt_manifest) and the
callbacks (salt3_staged_cpt_callbacks) into one resumable, milestone-capturing arm.

run_session trains the NEXT disjoint chunk window of a shared ceiling cache, advances a
per-arm manifest, snapshots milestones, logs continuous global-step metrics, and forks a
cooldown branch at reached table-milestones. Re-run it to continue (next window) or to resume
(same window, after a crash). One hardcoded INIT_NAME per run — swap the name for a new arm.

Reserved cooldown pool: the TAIL of the train split ([train_chunks - pool_len, train_chunks))
is held out from every trunk window, so the trunk budget curve is never contaminated by the
LR-annealing data and one ceiling cache still serves everything (disjoint by construction).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from salt3_staged_cpt_plot import plot_by_budget  # re-export so the notebook keeps one import

__all__ = ["run_session", "plot_by_budget"]


def _recover_current_model(arm_dir: Path) -> None:
    """Heal the rolling model after a crash inside the swap: if current_model vanished but a
    backup survives, promote the backup; drop a stale backup once current_model is intact."""
    cur, bak = arm_dir / "current_model", arm_dir / "current_model.bak"
    if not (cur / "model.safetensors").exists() and (bak / "model.safetensors").exists():
        shutil.rmtree(cur, ignore_errors=True)
        os.replace(bak, cur)
    elif (cur / "model.safetensors").exists() and bak.exists():
        shutil.rmtree(bak, ignore_errors=True)


def _atomic_save_model(trainer, tokenizer, dest: Path) -> None:
    """Save model+tokenizer+patched remote files to a temp dir, then swap it into place via
    two atomic renames (old -> .bak -> new -> drop .bak), so a reader never sees a half-write."""
    from salt3_common import copy_neobert_remote_files, ensure_dir

    dest = Path(dest)
    tmp = dest.parent / (dest.name + ".tmp")
    bak = dest.parent / (dest.name + ".bak")
    shutil.rmtree(tmp, ignore_errors=True)
    ensure_dir(tmp)
    trainer.save_model(tmp)
    tokenizer.save_pretrained(tmp)
    copy_neobert_remote_files(tmp)  # patched rotary/SwiGLU -> NaN-free reload
    if dest.exists():
        shutil.rmtree(bak, ignore_errors=True)
        os.replace(dest, bak)
    os.replace(tmp, dest)
    shutil.rmtree(bak, ignore_errors=True)


def run_session(init_name, session_docs, *, runs_root, init_root, dataset_cache, ceiling_docs,
                max_seq_len: int = 1024, peak_lr: float = 1e-4, warmup_steps: int | None = None,
                milestones_docs=(), table_milestones=(), eval_chunks: int = 1024,
                per_device_bs: int = 32, grad_accum: int = 16, seed: int = 42,
                eval_ratio: float = 0.02, mlm_probability: float = 0.20, mask_scheme: str = "mask_all",
                pool_len_chunks: int = 20_000, eval_steps: int = 200, save_steps: int = 200,
                device: str = "cuda") -> dict:
    import gc

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from salt3_common import (chunks_to_steps, docs_to_chunks, ensure_dir, make_mlm_collator,
                              make_mlm_datasets, make_neo_mlm_trainer_class, load_model_safe,
                              read_cache_meta, tokenizer_fingerprint, train_with_run_resume,
                              training_args)
    import salt3_staged_cpt_callbacks as cb
    import salt3_staged_cpt_manifest as mf
    import salt3_staged_schedule as sched

    eff_batch = per_device_bs * grad_accum
    arm_dir = ensure_dir(Path(runs_root) / init_name)
    base_init_dir = Path(init_root) / init_name / "model"
    current_model = arm_dir / "current_model"
    session_ckpt = arm_dir / "session_ckpt"
    metrics_path = arm_dir / "metrics.jsonl"
    opt_state_path = arm_dir / "optimizer_state.pt"
    _recover_current_model(arm_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_init_dir, trust_remote_code=True)
    meta = read_cache_meta(dataset_cache, tokenizer, ceiling_docs, max_seq_len, eval_ratio, seed)
    if meta is None:
        # surface the full cache key (docs+seq+tok_fp+eval_ratio+seed) so a silent drift vs the build call is visible
        raise FileNotFoundError(
            f"No ceiling cache_meta for docs={ceiling_docs:,} seq={max_seq_len} eval_ratio={eval_ratio} "
            f"seed={seed} tok={tokenizer_fingerprint(tokenizer)['short']} — build the ceiling cache "
            "first (notebook ceiling cell) with the SAME params before run_session.")
    ratio = meta["ratio_chunks_per_doc"]
    train_chunks_total = meta["train_chunks"]

    # reserve the cooldown pool as the train-split tail iff any table-milestone needs it
    pool_len = min(pool_len_chunks, max(1, train_chunks_total // 5)) if table_milestones else 0
    trunk_ceiling = train_chunks_total - pool_len
    if trunk_ceiling < 1:
        raise ValueError(f"cache too small ({train_chunks_total} train chunks) for pool {pool_len}")

    mpath = mf.manifest_path(runs_root, init_name)
    manifest = mf.load_manifest(mpath)
    if manifest is None:
        if warmup_steps is None:
            warmup_steps = sched.compute_warmup_steps(chunks_to_steps(trunk_ceiling, eff_batch))
        manifest = mf.init_manifest(init_name, tokenizer_fingerprint(tokenizer)["short"],
                                    peak_lr=peak_lr, warmup_steps=warmup_steps)
    else:
        warmup_steps = manifest["wsd"]["warmup_steps"]  # pinned at arm creation -> LR never drifts

    chunk_start = manifest["chunks_consumed"]
    if chunk_start >= trunk_ceiling:
        print(f"[{init_name}] trunk ceiling reached ({chunk_start:,}/{trunk_ceiling:,} chunks) — nothing to train")
        return {"status": "ceiling_reached", "manifest": manifest}
    chunk_end = min(chunk_start + docs_to_chunks(session_docs, ratio), trunk_ceiling)
    session_steps = chunks_to_steps(chunk_end - chunk_start, eff_batch)
    offset = manifest["global_step"]
    print(f"[{init_name}] window chunks [{chunk_start:,}, {chunk_end:,}) ~ {session_steps:,} steps "
          f"| global offset {offset:,} | warmup {warmup_steps}")

    train_ds, eval_ds = make_mlm_datasets(
        tokenizer=tokenizer, cache_dir=dataset_cache, num_examples=ceiling_docs,
        max_seq_len=max_seq_len, eval_ratio=eval_ratio, seed=seed,
        chunk_start=chunk_start, chunk_end=chunk_end, eval_chunks=eval_chunks)
    collator = sched.build_mlm_collator_compat(make_mlm_collator, tokenizer, mlm_probability, mask_scheme)

    base_for_model = current_model if (current_model / "model.safetensors").exists() else base_init_dir
    model = load_model_safe(base_for_model, model_cls=AutoModelForMaskedLM, device=device)
    optimizer = sched.build_wsd_optimizer(model, peak_lr=peak_lr)
    sched.load_optimizer_state(optimizer, opt_state_path)

    NeoMLMTrainer = make_neo_mlm_trainer_class()
    args = training_args(session_ckpt, max_steps=session_steps,
                         per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=grad_accum,
                         warmup_ratio=0.0, lr_scheduler_type="constant", eval_steps=eval_steps,
                         save_steps=save_steps, load_best_model_at_end=False)

    # union so a table-milestone always gets a stable snapshot + a step threshold even if the
    # caller forgot to also list it in milestones_docs
    all_milestones = sorted({int(x) for x in list(milestones_docs) + list(table_milestones)})
    milestone_steps = {m: chunks_to_steps(docs_to_chunks(m, ratio), eff_batch) for m in all_milestones}
    snapped = set(manifest.get("milestones_snapped", []))
    callbacks = [
        sched.make_wsd_lr_callback(peak_lr, warmup_steps, global_step_offset=offset),
        cb.make_staged_metrics_callback(metrics_path, global_step_offset=offset, ratio=ratio,
                                        seq_len=max_seq_len, eff_batch=eff_batch),
        cb.make_milestone_callback(milestone_steps, global_step_offset=offset, out_root=arm_dir,
                                   tokenizer=tokenizer, snapped=snapped),
    ]
    trainer = NeoMLMTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                            data_collator=collator, processing_class=tokenizer,
                            optimizers=(optimizer, None), callbacks=callbacks)
    result = train_with_run_resume(trainer, "resume")  # intra-session resume from session_ckpt

    # advance ONLY after a clean return (crash leaves manifest pre-advance; HF ckpt covers the gap)
    _atomic_save_model(trainer, tokenizer, current_model)
    sched.save_optimizer_state(optimizer, opt_state_path)
    manifest = mf.advance_manifest(
        manifest, new_chunks_consumed=chunk_end, global_step_delta=result.global_step,
        docs_consumed=int(chunk_end / ratio), tokens_seen=chunk_end * max_seq_len,
        window=(chunk_start, chunk_end), session_steps=session_steps)
    for tag in sorted(snapped):
        mf.mark_milestone(manifest, "milestones_snapped", int(tag))
    mf.save_manifest(mpath, manifest)
    shutil.rmtree(session_ckpt, ignore_errors=True)  # next window resumes clean, not this one

    del trainer, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # cooldown branches for reached table milestones (reserved pool, reused across milestones)
    cooled = set(manifest.get("milestones_cooled", []))
    reached = [int(m) for m in table_milestones
               if milestone_steps.get(int(m), 1 << 60) <= manifest["global_step"] and int(m) not in cooled]
    if reached and pool_len > 0:
        pool_train, pool_eval = make_mlm_datasets(
            tokenizer=tokenizer, cache_dir=dataset_cache, num_examples=ceiling_docs,
            max_seq_len=max_seq_len, eval_ratio=eval_ratio, seed=seed,
            chunk_start=trunk_ceiling, chunk_end=trunk_ceiling + pool_len, eval_chunks=eval_chunks)
        for m in reached:
            # fork THIS milestone's stable snapshot (a session may span several table
            # milestones, so current_model is not necessarily at budget m); cooldown length
            # scales with the milestone's own budget, not the session end.
            stable = arm_dir / f"milestone_{m}_stable"
            base = stable if (stable / "model.safetensors").exists() else current_model
            sched.run_cooldown_branch(
                base, cooldown_steps=sched.cooldown_steps_for(milestone_steps[int(m)]),
                peak_lr=peak_lr, pool_train_ds=pool_train, eval_ds=pool_eval, tokenizer=tokenizer,
                out_dir=arm_dir / f"milestone_{m}_cooled", device=device, mlm_probability=mlm_probability,
                mask_scheme=mask_scheme,
                snapshot_extra={"docs": m, "global_step": milestone_steps[int(m)], "tag": m})
            mf.mark_milestone(manifest, "milestones_cooled", m)
            mf.save_manifest(mpath, manifest)

    print(f"[{init_name}] done: global_step={manifest['global_step']:,} chunks={manifest['chunks_consumed']:,} "
          f"snapped={manifest['milestones_snapped']} cooled={manifest['milestones_cooled']}")
    return {"status": "trained", "manifest": manifest, "window": (chunk_start, chunk_end),
            "session_steps": session_steps, "result_global_step": result.global_step}
