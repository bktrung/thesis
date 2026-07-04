"""CPU smoke tests for the staged-WSD-CPT harness — mirrors salt3_patch_cpu_checks.py.

Tier A (always, torch-free): WSD LR shape + no-rewarm, warmup clamp, cooldown cosine->0 + cap,
manifest round-trip + idempotent advance, chunk-window disjointness/continuity + beyond-cache
clamp, constant eval slice, docs<->chunks converters, tokenizer-fingerprint reuse-or-raise.
Verified the way the patch checks verify patch fns: AST-extract the torch-free helpers from
salt3_common (skipping its top-level ``import torch``); import the torch-free modules directly.

Tier B (needs torch+transformers): live WSDLRCallback no-rewarm on a real optimizer, and
optimizer-state carry round-trip. SKIPPED (not failed) when the ML stack is absent, so this
passes on a torch-free dev box and fully exercises on Colab.

The full run_session continue/resume + milestone-reload integration runs as the notebook
dry-run (16_staged_wsd_cpt.ipynb, tiny CEILING_DOCS) — it needs a real init dir + datasets.

Usage: python3 salt3_staged_cpt_checks.py
"""
import ast
import sys
import tempfile
from pathlib import Path

import salt3_staged_cpt_manifest as mf
import salt3_staged_schedule as sched

CODE_DIR = Path(__file__).resolve().parent
FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def skip(name: str, why: str) -> None:
    print(f"  SKIP  {name}  ({why})")
    SKIPPED.append(name)


def extract_from_common(names: set[str]) -> dict:
    """Exec only the named torch-free defs from salt3_common.py (its top-level ``import torch``
    is never executed because we compile node-by-node)."""
    tree = ast.parse((CODE_DIR / "salt3_common.py").read_text(encoding="utf-8"))
    ns: dict = {"Path": Path}
    for node in tree.body:
        if getattr(node, "name", None) in names:
            exec(compile(ast.Module([node], []), "salt3_common.py", "exec"), ns)
    missing = names - set(ns)
    assert not missing, f"missing from salt3_common: {missing}"
    return ns


def undefined_loads(func_name: str) -> set[str]:
    """Names a function in salt3_common LOADs but never binds (param/assignment/import/
    comprehension target) and that aren't module globals or builtins. Catches the class of
    bug where a refactor deletes a local but leaves a use (e.g. a dangling ``cache_key``) on a
    branch the slice-only smoke tests never execute."""
    import builtins

    tree = ast.parse((CODE_DIR / "salt3_common.py").read_text(encoding="utf-8"))
    mod = set(dir(builtins))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mod.add(n.name)
        elif isinstance(n, ast.Assign):
            mod.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            mod.update((a.asname or a.name).split(".")[0] for a in n.names)
    func = next(x for x in ast.walk(tree)
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == func_name)
    bound: set[str] = set()
    loads: set[str] = set()

    def add_args(a):
        for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
            bound.add(x.arg)
        if a.vararg:
            bound.add(a.vararg.arg)
        if a.kwarg:
            bound.add(a.kwarg.arg)

    add_args(func.args)
    for node in ast.walk(func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func:
            add_args(node.args)
            bound.add(node.name)
        elif isinstance(node, ast.Lambda):
            add_args(node.args)
        elif isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, ast.Store) else loads).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
    return loads - bound - mod


class FakeDS(list):
    """Dataset stand-in: len() + .select(range)->FakeDS; elements ARE the global chunk indices
    so a window's disjointness/continuity is directly inspectable without datasets installed."""

    def select(self, indices) -> "FakeDS":
        return FakeDS(self[i] for i in indices)


def tier_a() -> None:
    peak, warm = 1e-4, 100

    print("[A] no-undefined-names guard (cache-MISS build path)")
    for fn in ("make_mlm_datasets", "_select_datasets", "_dataset_cache_dir"):
        missing = undefined_loads(fn)
        check(f"{fn}: no undefined names", missing == set(), str(missing) if missing else "")

    print("[A] WSD LR shape + no-rewarm")
    check("ramp 0->peak over warmup",
          abs(sched.wsd_lr_at(0, peak_lr=peak, warmup_steps=warm)) < 1e-12
          and abs(sched.wsd_lr_at(50, peak_lr=peak, warmup_steps=warm) - peak * 0.5) < 1e-12)
    check("flat == peak for all global >= warmup",
          all(abs(sched.wsd_lr_at(g, peak_lr=peak, warmup_steps=warm) - peak) < 1e-12
              for g in (warm, warm + 1, warm + 5000)))
    check("no re-warm: offset>=warmup -> local 0 is peak",
          abs(sched.wsd_lr_at(warm + 0, peak_lr=peak, warmup_steps=warm) - peak) < 1e-12)
    check("first session ramps: offset 0 -> local 1 < peak",
          sched.wsd_lr_at(1, peak_lr=peak, warmup_steps=warm) < peak)

    print("[A] warmup clamp + cooldown")
    check("warmup clamps to [20,500]",
          sched.compute_warmup_steps(10) == 20 and sched.compute_warmup_steps(10 ** 9) == 500
          and sched.compute_warmup_steps(5000) == 100)
    dry = 5000 // 2  # 5k-doc dry-run at a small CPU eff-batch -> plenty of steps
    check("warmup < dry-run total steps", sched.compute_warmup_steps(dry) < dry,
          f"{sched.compute_warmup_steps(dry)} < {dry}")
    check("cooldown cosine peak->0",
          abs(sched.cooldown_lr_at(0, peak_lr=peak, cooldown_steps=200) - peak) < 1e-12
          and sched.cooldown_lr_at(200, peak_lr=peak, cooldown_steps=200) < 1e-9)
    check("cooldown step cap respected",
          sched.cooldown_steps_for(10 ** 7) == 1000 and sched.cooldown_steps_for(100) == 10)

    print("[A] manifest round-trip + idempotent advance")
    with tempfile.TemporaryDirectory() as td:
        mpath = mf.manifest_path(td, "armX")
        m = mf.init_manifest("armX", "fp123", peak_lr=peak, warmup_steps=50)
        mf.save_manifest(mpath, m)
        check("round-trip equal", mf.load_manifest(mpath) == m)
        m = mf.advance_manifest(m, new_chunks_consumed=1000, global_step_delta=2, docs_consumed=1000,
                                tokens_seen=1000 * 1024, window=(0, 1000), session_steps=2)
        m = mf.advance_manifest(m, new_chunks_consumed=2000, global_step_delta=2, docs_consumed=2000,
                                tokens_seen=2000 * 1024, window=(1000, 2000), session_steps=2)
        check("two sessions: chunks doubled, global summed, 2 sessions",
              m["chunks_consumed"] == 2000 and m["global_step"] == 4 and len(m["sessions"]) == 2)
        mf.mark_milestone(m, "milestones_snapped", 100)
        mf.mark_milestone(m, "milestones_snapped", 100)
        check("mark_milestone idempotent", m["milestones_snapped"] == [100])

    print("[A] chunk-window disjointness/continuity + constant eval")
    ns = extract_from_common({"_slice_chunks", "_select_datasets", "docs_to_chunks", "chunks_to_steps"})
    sel = ns["_select_datasets"]
    t1, _ = sel(FakeDS(range(5000)), FakeDS(range(200)), num_chunks=None, eval_ratio=0.02,
                chunk_start=0, chunk_end=1000, eval_chunks=64)
    t2, _ = sel(FakeDS(range(5000)), FakeDS(range(200)), num_chunks=None, eval_ratio=0.02,
                chunk_start=1000, chunk_end=2000, eval_chunks=64)
    check("windows contiguous", list(t1) == list(range(0, 1000)) and list(t2) == list(range(1000, 2000)))
    check("windows disjoint", set(t1).isdisjoint(set(t2)))
    _, e_big = sel(FakeDS(range(5000)), FakeDS(range(500)), num_chunks=None, eval_ratio=0.02,
                   chunk_start=0, chunk_end=10, eval_chunks=128)
    _, e_small = sel(FakeDS(range(2000)), FakeDS(range(500)), num_chunks=None, eval_ratio=0.02,
                     chunk_start=0, chunk_end=10, eval_chunks=128)
    check("eval slice constant across cache sizes", len(e_big) == 128 and len(e_small) == 128)
    tcl, _ = sel(FakeDS(range(100)), FakeDS(range(10)), num_chunks=None, eval_ratio=0.02,
                 chunk_start=50, chunk_end=9999, eval_chunks=5)
    check("window clamps to cache length", list(tcl) == list(range(50, 100)))
    check("docs<->chunks converters",
          ns["docs_to_chunks"](1000, 1.5) == 1500 and ns["chunks_to_steps"](1000, 512) == 2)

    print("[A] tokenizer-fingerprint reuse-or-raise")
    asn = extract_from_common({"assert_shared_tokenizer"})
    asn["tokenizer_fingerprint"] = lambda tok: {"short": tok}  # stub: the string IS its fp
    assert_shared = asn["assert_shared_tokenizer"]
    check("shared fp returns the common short", assert_shared(["abc", "abc", "abc"]) == "abc")
    raised = False
    try:
        assert_shared(["abc", "xyz"])
    except ValueError:
        raised = True
    check("mismatched fp raises", raised)


def tier_b() -> None:
    try:
        import torch
        from transformers import TrainerCallback  # noqa: F401  (proves the stack is importable)
    except Exception as exc:
        skip("Tier B (live callback + optimizer carry)", f"ML stack absent: {type(exc).__name__}")
        return

    peak, warm = 1e-4, 100

    class _State:
        def __init__(self, g):
            self.global_step = g

    class _Opt:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

    print("[B] live WSDLRCallback no-rewarm on a real optimizer")
    resumed = sched.make_wsd_lr_callback(peak, warm, global_step_offset=warm)
    opt = _Opt()
    resumed.on_step_begin(None, _State(0), None, optimizer=opt)
    check("resume: callback sets lr==peak at local step 0", abs(opt.param_groups[0]["lr"] - peak) < 1e-12)
    fresh = sched.make_wsd_lr_callback(peak, warm, global_step_offset=0)
    opt2 = _Opt()
    fresh.on_step_begin(None, _State(1), None, optimizer=opt2)
    check("fresh: callback ramps lr (<peak) at local step 1", opt2.param_groups[0]["lr"] < peak)

    print("[B] optimizer-state carry round-trip")
    lin = torch.nn.Linear(4, 4)
    opt = sched.build_wsd_optimizer(lin, peak_lr=peak)
    lin(torch.randn(2, 4)).sum().backward()
    opt.step()  # populate Adam moments
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "opt.pt"
        sched.save_optimizer_state(opt, p)
        lin2 = torch.nn.Linear(4, 4)
        opt2 = sched.build_wsd_optimizer(lin2, peak_lr=peak)
        loaded = sched.load_optimizer_state(opt2, p)
        s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
        eq = loaded and all(torch.allclose(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)
        check("Adam moments equal after reload", bool(eq))


def main() -> None:
    print("=" * 60)
    print("staged-WSD-CPT CPU smoke checks")
    print("=" * 60)
    tier_a()
    tier_b()
    print("=" * 60)
    if FAILURES:
        print(f"❌ {len(FAILURES)} FAILED: {FAILURES}" + (f" | {len(SKIPPED)} skipped" if SKIPPED else ""))
        sys.exit(1)
    print(f"✅ all checks passed" + (f" ({len(SKIPPED)} Tier-B skipped: ML stack absent)" if SKIPPED else ""))


if __name__ == "__main__":
    main()
