"""Per-arm staged-CPT manifest: the inter-session source of truth for two-level resume.

The manifest records how far an arm has trained in GLOBAL units (steps, chunks, docs, tokens)
plus which milestones are snapped/cooled, so re-running ``run_session`` continues to the NEXT
disjoint window instead of re-reading doc 0. It is advanced ONLY after a clean train return;
a crash leaves it pre-advance and the intra-session HF checkpoint covers the gap (no double
count, no data re-read). Saved atomically (temp + os.replace) so a mid-write crash can't
corrupt the resume state.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def manifest_path(runs_root, init_name: str) -> Path:
    return Path(runs_root) / init_name / "manifest.json"


def init_manifest(init_name: str, init_fp: str, *, peak_lr: float, warmup_steps: int) -> dict[str, Any]:
    return {
        "init_name": init_name,
        "init_fp": init_fp,
        "global_step": 0,
        "chunks_consumed": 0,
        "docs_consumed": 0,
        "tokens_seen": 0,
        "milestones_snapped": [],
        "milestones_cooled": [],
        "wsd": {"peak_lr": peak_lr, "warmup_steps": warmup_steps},
        "sessions": [],
    }


def load_manifest(path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem


def advance_manifest(manifest: dict[str, Any], *, new_chunks_consumed: int, global_step_delta: int,
                     docs_consumed: int, tokens_seen: int, window: tuple[int, int],
                     session_steps: int) -> dict[str, Any]:
    """Apply one finished session's progress. Idempotency is the caller's job: it advances
    only after a clean return, so a re-run after a crash recomputes the SAME window."""
    manifest["global_step"] += int(global_step_delta)
    manifest["chunks_consumed"] = int(new_chunks_consumed)
    manifest["docs_consumed"] = int(docs_consumed)
    manifest["tokens_seen"] = int(tokens_seen)
    manifest["sessions"].append({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": [int(window[0]), int(window[1])],
        "steps": int(session_steps),
        "global_step": manifest["global_step"],
    })
    return manifest


def mark_milestone(manifest: dict[str, Any], key: str, milestone_docs: int) -> None:
    """Record a snapped/cooled milestone (``key`` is 'milestones_snapped' or 'milestones_cooled').
    Sorted + de-duplicated so repeated reconciliation passes are idempotent."""
    bucket = manifest.setdefault(key, [])
    if milestone_docs not in bucket:
        bucket.append(milestone_docs)
        bucket.sort()
