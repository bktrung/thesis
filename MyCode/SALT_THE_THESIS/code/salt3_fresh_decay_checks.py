"""CPU smoke checks for the fresh-data decay branch — torch-free.

Verifies the 1-sqrt cooldown shape (endpoints, faster-early, monotonicity, clamp), the shape
registry, and the pure decay-window planner: single-pass/no-repeat sizing, contiguity with the
trunk, clamp-to-unseen (reuse-first), honest achieved-fraction, ratio handling, and the
no-unseen-data guard. ``plan_decay_window`` is AST-extracted from salt3_fresh_decay so its
torch-pulling imports never execute (same trick as salt3_staged_cpt_checks).

Usage: python3 salt3_fresh_decay_checks.py
"""
import ast
import math
from pathlib import Path

import salt3_staged_schedule as sched

CODE_DIR = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _extract(filename: str, names: set[str], seed_ns: dict | None = None) -> dict:
    """Exec only the named torch-free defs from a code file (its torch-pulling imports never run)."""
    tree = ast.parse((CODE_DIR / filename).read_text(encoding="utf-8"))
    ns: dict = dict(seed_ns or {})
    ns.setdefault("math", math)
    for node in tree.body:
        if getattr(node, "name", None) in names:
            exec(compile(ast.Module([node], []), filename, "exec"), ns)
    missing = names - set(ns)
    assert not missing, f"missing from {filename}: {missing}"
    return ns


_conv = _extract("salt3_common.py", {"docs_to_chunks", "chunks_to_steps"})
plan_decay_window = _extract("salt3_fresh_decay.py", {"plan_decay_window"}, _conv)["plan_decay_window"]

print("Fresh-data decay CPU checks")
print("─" * 60)

# ── 1-sqrt shape ─────────────────────────────────────────────────────────────
PEAK, CD = 1e-4, 1000
check("onesqrt LR(0) == peak", abs(sched.onesqrt_lr_at(0, peak_lr=PEAK, cooldown_steps=CD) - PEAK) < 1e-15)
check("onesqrt LR(end) == 0", abs(sched.onesqrt_lr_at(CD, peak_lr=PEAK, cooldown_steps=CD)) < 1e-15)
mid = sched.onesqrt_lr_at(CD // 2, peak_lr=PEAK, cooldown_steps=CD)
cos_mid = sched.cooldown_lr_at(CD // 2, peak_lr=PEAK, cooldown_steps=CD)
check("onesqrt drops faster early than cosine", mid < cos_mid, f"{mid:.2e} < {cos_mid:.2e}")
seq = [sched.onesqrt_lr_at(s, peak_lr=PEAK, cooldown_steps=CD) for s in range(0, CD + 1, 25)]
check("onesqrt monotone non-increasing", all(a >= b - 1e-18 for a, b in zip(seq, seq[1:])))
check("onesqrt clamps past end == 0", sched.onesqrt_lr_at(CD * 2, peak_lr=PEAK, cooldown_steps=CD) == 0.0)
check("onesqrt cooldown_steps<=0 -> 0", sched.onesqrt_lr_at(5, peak_lr=PEAK, cooldown_steps=0) == 0.0)
check("shape registry == {cosine,1-sqrt}", set(sched.COOLDOWN_SHAPES) == {"cosine", "1-sqrt"})
check("unknown shape not in registry", sched.COOLDOWN_SHAPES.get("nope") is None)

# ── decay-window planner ─────────────────────────────────────────────────────
EFF = 512
# A: plenty of unseen -> full ~20%, single pass
a = plan_decay_window(chunks_consumed=4_000_000, train_chunks=6_000_000, ratio=1.0,
                      eff_batch=EFF, target_total_docs=5_000_000, decay_frac=0.20)
check("A start == chunks_consumed (contiguity)", a["chunk_start"] == 4_000_000)
check("A no-repeat (chunks == steps*eff)", a["decay_chunks"] == a["decay_steps"] * EFF)
check("A window within unseen", a["chunk_end"] - a["chunk_start"] <= a["unseen_chunks"])
check("A decay_steps == round(0.2*total)", a["decay_steps"] == round(0.20 * a["total_steps"]))
check("A achieved ~= 0.20", abs(a["achieved_frac"] - 0.20) < 0.02, f"{a['achieved_frac']}")

# B: tight cache (parked at 4M, ~0.9M unseen) -> clamp to ~18%, still single pass
b = plan_decay_window(chunks_consumed=4_000_000, train_chunks=4_900_000, ratio=1.0,
                      eff_batch=EFF, target_total_docs=5_000_000, decay_frac=0.20)
check("B clamps to unseen", b["decay_chunks"] <= b["unseen_chunks"])
check("B no-repeat", b["decay_chunks"] == b["decay_steps"] * EFF)
check("B achieved < 0.20 (clamped)", b["achieved_frac"] < 0.20, f"{b['achieved_frac']}")
check("B achieved within WSD band (>=0.10)", b["achieved_frac"] >= 0.10, f"{b['achieved_frac']}")

# C: no unseen room -> zero steps (caller must extend the cache)
c = plan_decay_window(chunks_consumed=4_900_000, train_chunks=4_900_000, ratio=1.0,
                      eff_batch=EFF, target_total_docs=5_000_000, decay_frac=0.20)
check("C zero steps when no unseen", c["decay_steps"] == 0)

# D: ratio != 1 honored end-to-end
d = plan_decay_window(chunks_consumed=100, train_chunks=10_000, ratio=2.0,
                      eff_batch=10, target_total_docs=1000, decay_frac=0.50)
check("D ratio honored (steps)", d["decay_steps"] == 100, f"{d['decay_steps']}")

print("─" * 60)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
    raise SystemExit(1)
print("All fresh-data decay checks passed.")
