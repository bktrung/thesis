"""CPU smoke checks for the 4k context-extension modules — torch/datasets-free.

Verifies the pure planner (single-pass/no-repeat sizing, contiguity with the fork point,
clamp-to-unseen reuse-first, honest achieved-token budget, zero-unseen guard), the regroup core
(byte-exact 4x1024 -> 4096 reconstruction on an arange synthetic cache), the PPPL masking + math
(valid non-special positions, exp(mean CE)), and the WSD warmup->constant LR plateau.

Pure defs are AST-extracted from salt3_ctx4096 / salt3_pppl_curve so their torch/datasets-pulling
imports never execute (same trick as salt3_fresh_decay_checks).

Usage: python3 salt3_ctx4096_checks.py
"""
import ast
import math
import random
from pathlib import Path

import salt3_staged_schedule as sched

CODE_DIR = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _extract(filename: str, names: set[str], seed_ns: dict | None = None) -> dict:
    """Exec only the named torch-free defs from a code file (its heavy imports never run)."""
    tree = ast.parse((CODE_DIR / filename).read_text(encoding="utf-8"))
    ns: dict = dict(seed_ns or {})
    ns.setdefault("math", math)
    ns.setdefault("random", random)
    for node in tree.body:
        if getattr(node, "name", None) in names:
            exec(compile(ast.Module([node], []), filename, "exec"), ns)
    missing = names - set(ns)
    assert not missing, f"missing from {filename}: {missing}"
    return ns


_ctx = _extract("salt3_ctx4096.py", {"plan_ctx4096_window", "_regroup_rows"}, {"BASE_SEQ_LEN": 1024})
plan_ctx4096_window = _ctx["plan_ctx4096_window"]
_regroup_rows = _ctx["_regroup_rows"]
_pppl = _extract("salt3_pppl_curve.py", {"sample_mask_positions", "pppl_from_ce"})
sample_mask_positions = _pppl["sample_mask_positions"]
pppl_from_ce = _pppl["pppl_from_ce"]

print("4k context-extension CPU checks")
print("─" * 60)

# ── planner ──────────────────────────────────────────────────────────────────
GROUP, EFF = 4, 128
STEP_1024 = GROUP * EFF  # 1024-chunks consumed per step

# A: plenty of unseen -> hits the ~200M target, single pass, not clamped
a = plan_ctx4096_window(chunks_consumed=1_000_000, train_chunks_1024=3_000_000, group=GROUP,
                        eff_batch=EFF, target_tokens=200_000_000)
check("A start == chunks_consumed (contiguity)", a["chunk_start"] == 1_000_000)
check("A steps == want_steps (not clamped)", a["steps"] == a["want_steps"], f"{a['steps']}")
check("A no-repeat (n_1024 == n_4096*group)", a["n_1024_chunks"] == a["n_4096_chunks"] * GROUP)
check("A n_1024 multiple of step (whole steps)", a["n_1024_chunks"] % STEP_1024 == 0)
check("A window within unseen tail", a["n_1024_chunks"] <= a["unseen_1024_chunks"])
check("A chunk_end == start + n_1024", a["chunk_end"] == a["chunk_start"] + a["n_1024_chunks"])
check("A seq_len == 4096", a["seq_len"] == 4096)
check("A achieved ~ target (>=0.99)", a["achieved_frac"] >= 0.99, f"{a['achieved_frac']}")

# B: tight unseen tail -> clamp below target, still single pass no-repeat
b = plan_ctx4096_window(chunks_consumed=2_900_000, train_chunks_1024=3_000_000, group=GROUP,
                        eff_batch=EFF, target_tokens=200_000_000)
check("B clamps below want_steps", b["steps"] < b["want_steps"], f"{b['steps']}<{b['want_steps']}")
check("B within unseen tail", b["n_1024_chunks"] <= b["unseen_1024_chunks"])
check("B no-repeat", b["n_1024_chunks"] == b["n_4096_chunks"] * GROUP)
check("B achieved < 1.0 (clamped)", b["achieved_frac"] < 1.0, f"{b['achieved_frac']}")

# C: no unseen room -> zero steps (caller must extend the cache)
c = plan_ctx4096_window(chunks_consumed=3_000_000, train_chunks_1024=3_000_000, group=GROUP,
                        eff_batch=EFF, target_tokens=200_000_000)
check("C zero steps when no unseen", c["steps"] == 0 and c["n_4096_chunks"] == 0)

# D: unseen tail smaller than one step's worth (< group*eff_batch 1024-chunks) -> floors to 0
d = plan_ctx4096_window(chunks_consumed=2_999_600, train_chunks_1024=3_000_000, group=GROUP,
                        eff_batch=EFF, target_tokens=200_000_000)
check("D floors sub-step tail to 0", d["steps"] == 0,
      f"unseen={d['unseen_1024_chunks']} < step={STEP_1024}")

# ── regroup contiguity (arange synthetic cache: byte-checkable reconstruction) ─
N = 13  # not a multiple of group -> exercises the dropped-remainder path
rows = [list(range(i * 1024, (i + 1) * 1024)) for i in range(N)]
merged = list(_regroup_rows(iter(rows), GROUP))
check("regroup count == floor(N/group)", len(merged) == N // GROUP, f"{len(merged)}")
check("regroup rows all len 4096", all(len(r) == 4096 for r in merged))
exact = all(merged[g] == list(range(g * 4096, g * 4096 + 4096)) for g in range(len(merged)))
check("regroup byte-exact contiguous 4x1024==4096", exact)
check("regroup drops ragged tail (< group)", N - len(merged) * GROUP == N % GROUP)

# ── PPPL masking + math ──────────────────────────────────────────────────────
# specials at 0,1 and last position; the rest maskable
sm = [True, True] + [False] * 20 + [True]
pos = sample_mask_positions(sm, n_masks=8, seed=42)
check("mask excludes all specials", all(not sm[i] for i in pos))
check("mask count == n_masks (enough candidates)", len(pos) == 8, f"{len(pos)}")
check("mask sorted", pos == sorted(pos))
check("mask deterministic in seed", pos == sample_mask_positions(sm, n_masks=8, seed=42))
short = sample_mask_positions([False, False, False, True], n_masks=8, seed=1)
check("mask returns all when candidates < n_masks", len(short) == 3 and short == [0, 1, 2])
check("pppl exp(mean CE): all-zero -> 1.0", abs(pppl_from_ce([0.0, 0.0, 0.0]) - 1.0) < 1e-12)
check("pppl exp(mean CE): log2 -> 2.0", abs(pppl_from_ce([math.log(2)] * 5) - 2.0) < 1e-12)
check("pppl empty -> nan", math.isnan(pppl_from_ce([])))

# ── WSD warmup-then-constant LR plateau (the stage-2 constant LR) ─────────────
PEAK, WU = 1e-5, 50
check("wsd LR(0) == 0", sched.wsd_lr_at(0, peak_lr=PEAK, warmup_steps=WU) == 0.0)
check("wsd LR(mid) == peak/2", abs(sched.wsd_lr_at(WU // 2, peak_lr=PEAK, warmup_steps=WU) - PEAK / 2) < 1e-18)
check("wsd LR(warmup) == peak", abs(sched.wsd_lr_at(WU, peak_lr=PEAK, warmup_steps=WU) - PEAK) < 1e-18)
check("wsd LR flat after warmup", sched.wsd_lr_at(WU * 20, peak_lr=PEAK, warmup_steps=WU) == PEAK)
check("wsd warmup<=0 -> peak from step 0", sched.wsd_lr_at(0, peak_lr=PEAK, warmup_steps=0) == PEAK)

print("─" * 60)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
    raise SystemExit(1)
print("All 4k context-extension checks passed.")
