"""Trainer callbacks for staged CPT, both keyed on the GLOBAL step (offset + local) so
they behave continuously across sessions:

- milestone callback: generalizes EarlySnapshotCallback (single fraction) to an arbitrary
  ``{docs: global_step}`` map, saving an idempotent reloadable stable snapshot at the first
  crossing of each milestone.
- metrics callback: like JsonlMetricsCallback but logs the GLOBAL step and injects budget
  fields (chunks/docs/tokens seen) + the ACTUAL applied WSD lr (the HF-logged learning_rate
  reflects the inert constant placeholder, not the warmup ramp), all into one appended file.

Factory functions (not module-level classes) keep transformers import lazy, so the pure WSD
helpers stay importable on a torch-free box.
"""
from __future__ import annotations

import time
from pathlib import Path


def make_milestone_callback(milestone_steps: dict[int, int], *, global_step_offset: int,
                            out_root, tokenizer, snapped: set):
    """Save a stable snapshot at the first GLOBAL step that crosses each milestone.

    ``milestone_steps`` maps a milestone's docs tag -> its global-step threshold. ``snapped``
    is shared with the runner so it learns post-session which milestones fired. Idempotent:
    skips a milestone whose dir already holds weights (survives a resumed run)."""
    from transformers import TrainerCallback

    from salt3_common import copy_neobert_remote_files, ensure_dir, write_json

    out_root = Path(out_root)

    class _Milestone(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            g = global_step_offset + state.global_step
            for tag, threshold in milestone_steps.items():
                if tag in snapped or g < threshold:
                    continue
                snap_dir = out_root / f"milestone_{tag}_stable"
                if not (snap_dir / "model.safetensors").exists():
                    ensure_dir(snap_dir)
                    kwargs["model"].save_pretrained(snap_dir)
                    tokenizer.save_pretrained(snap_dir)
                    copy_neobert_remote_files(snap_dir)  # patched rotary/SwiGLU -> NaN-free reload
                    write_json(snap_dir / "snapshot_info.json",
                               {"kind": "stable", "docs": int(tag), "global_step": int(g)})
                    print(f"  stable snapshot @ global_step {g:,} (>= {tag:,} docs) -> {snap_dir}")
                snapped.add(tag)
            return control

    return _Milestone()


def make_staged_metrics_callback(metrics_path, *, global_step_offset: int, ratio: float,
                                 seq_len: int, eff_batch: int, session_tag: str | None = None):
    """Append one row per HF log to a single continuous ``metrics.jsonl``. ``step`` is the
    GLOBAL step so curves are continuous across sessions; budget fields let the plot slice by
    docs/tokens. A restarted session re-logs steps an abandoned one wrote — latest-per-step
    dedup at plot time keeps the live trajectory (same scheme as plot_training_curves)."""
    from transformers import TrainerCallback

    from salt3_common import append_jsonl

    tag = session_tag or time.strftime("%Y%m%d-%H%M%S")
    metrics_path = Path(metrics_path)

    class _Metrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            g = global_step_offset + state.global_step
            chunks = g * eff_batch
            record = {
                "step": int(g),
                "local_step": int(state.global_step),
                "epoch": None if state.epoch is None else float(state.epoch),
                "timestamp": time.time(),
                "session": tag,
                "chunks_seen": int(chunks),
                "docs_seen": int(chunks / ratio) if ratio else int(chunks),
                "tokens_seen": int(chunks * seq_len),
            }
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                record["wsd_lr"] = float(optimizer.param_groups[0]["lr"])
            record.update({k: v for k, v in logs.items()
                           if isinstance(v, (int, float, str, bool)) or v is None})
            record["step"] = int(g)  # never let a stray logs['step'] override the global step
            append_jsonl(metrics_path, record)
            return control

    return _Metrics()
