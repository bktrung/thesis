"""CPU-only verification of the NeoBERT NaN fixes in salt3_common.py — no torch needed.

Checks, against the REAL upstream NeoBERT remote-code files:
  1. _patch_rotary_py_complex: applies cleanly, output parses, idempotent,
     no complex math left in apply_rotary_emb.
  2. _patch_model_py_xformers: applies cleanly, output parses, idempotent,
     no xformers import left, w12/w3 param names preserved.
  3. Upgrade path: model.py already carrying the OLD try/except patch (what the
     saved init artifacts on Drive contain) upgrades to the pure-SwiGLU patch.
  4. Numpy oracle: the real-valued rotary formula emitted by the patch is
     numerically identical to the original complex-multiply rotary, for both
     freqs_cis shapes model.py produces ((1,seq,hd/2) and (batch,seq,hd/2)).

Usage: python3 salt3_patch_cpu_checks.py <upstream_model.py> <upstream_rotary.py>
"""
import ast
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load_patch_functions(source_path: Path, wanted: set | None = None) -> dict:
    """Exec only the patch functions + markers from salt3_common.py (skips torch imports)."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = wanted or {"_patch_model_py_xformers", "_patch_rotary_py_complex",
                        "_PURE_SWIGLU_MARKER", "_REAL_ROTARY_MARKER"}
    ns = {"Path": Path}
    for node in tree.body:
        name = getattr(node, "name", None) or (
            node.targets[0].id if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name) else None)
        if name in wanted:
            exec(compile(ast.Module([node], []), str(source_path), "exec"), ns)
    missing = wanted - set(ns)
    assert not missing, f"missing from {source_path}: {missing}"
    return ns


def parses(path: Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        return True
    except SyntaxError as e:
        print(f"      SyntaxError: {e}")
        return False


def apply_rotary_numpy_complex(xq, fc):
    """Original NeoBERT rotary semantics: view_as_complex pairs consecutive elements,
    freqs_cis (B,seq,hd/2) gets heads dim unsqueezed at axis 2, complex multiply."""
    xc = xq[..., 0::2] + 1j * xq[..., 1::2]          # (bs, seq, heads, hd/2)
    out = xc * fc[:, :, None, :]                      # broadcast (B, seq, 1, hd/2)
    res = np.empty_like(xq)
    res[..., 0::2] = out.real
    res[..., 1::2] = out.imag
    return res


def apply_rotary_numpy_real(xq, fc):
    """Mirror of the patched real-valued apply_rotary_emb, op by op, in numpy."""
    assert fc.shape[1:] == (xq.shape[1], xq.shape[-1] // 2), fc.shape
    cos = fc.real[:, :, None, :]                      # unsqueeze(2)
    sin = fc.imag[:, :, None, :]
    xf = xq.reshape(*xq.shape[:-1], -1, 2)
    a, b = xf[..., 0], xf[..., 1]
    out = np.stack([a * cos - b * sin, a * sin + b * cos], axis=-1)
    return out.reshape(*out.shape[:3], -1)            # flatten(3)


def main():
    model_src, rotary_src = Path(sys.argv[1]), Path(sys.argv[2])
    fns = load_patch_functions(CODE_DIR / "salt3_common.py")
    old_fns = load_patch_functions(Path("/tmp/salt3_common_old.py"),
                                   wanted={"_patch_model_py_xformers"}) \
        if Path("/tmp/salt3_common_old.py").exists() else None

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        print("[1] rotary.py patch")
        rp = td / "rotary.py"
        shutil.copy(rotary_src, rp)
        fns["_patch_rotary_py_complex"](rp)
        text = rp.read_text(encoding="utf-8")
        check("output parses", parses(rp))
        check("marker present", fns["_REAL_ROTARY_MARKER"] in text)
        body = text[text.find("def apply_rotary_emb("):]
        check("no complex math in apply_rotary_emb",
              all(k not in body for k in ("view_as_complex", "view_as_real", "reshape_for_broadcast(")))
        check("precompute_freqs_cis preserved", "def precompute_freqs_cis(" in text)
        before = text
        fns["_patch_rotary_py_complex"](rp)
        check("idempotent", rp.read_text(encoding="utf-8") == before)

        print("[2] model.py patch (fresh upstream)")
        mp = td / "model.py"
        shutil.copy(model_src, mp)
        fns["_patch_model_py_xformers"](mp)
        text = mp.read_text(encoding="utf-8")
        check("output parses", parses(mp))
        check("marker present", fns["_PURE_SWIGLU_MARKER"] in text)
        check("no xformers import left", "xformers" not in text)
        check("w12/w3 param names preserved", "self.w12" in text and "self.w3" in text)
        before = text
        fns["_patch_model_py_xformers"](mp)
        check("idempotent", mp.read_text(encoding="utf-8") == before)

        print("[3] upgrade path: old try/except patch -> pure SwiGLU")
        if old_fns is None:
            check("old salt3_common available", False, "/tmp/salt3_common_old.py missing")
        else:
            mp2 = td / "model_old_patched.py"
            shutil.copy(model_src, mp2)
            old_fns["_patch_model_py_xformers"](mp2)       # what Drive artifacts contain
            check("old patch applied (try/except present)",
                  "try:\n    from xformers.ops import SwiGLU" in mp2.read_text(encoding="utf-8"))
            fns["_patch_model_py_xformers"](mp2)           # the in-place upgrade
            text = mp2.read_text(encoding="utf-8")
            check("upgraded output parses", parses(mp2))
            check("upgraded to pure marker", fns["_PURE_SWIGLU_MARKER"] in text)
            check("no xformers import left after upgrade", "xformers" not in text)

    print("[4] numpy oracle: real-valued rotary == complex rotary")
    rng = np.random.default_rng(0)
    bs, seq, heads, hd = 2, 7, 10, 64
    theta = 10000.0 ** (-np.arange(0, hd, 2) / hd)
    ang = np.outer(np.arange(32), theta)[:seq]        # precompute_freqs_cis[:seq]
    fc1 = np.exp(1j * ang)[None]                      # (1, seq, hd/2) default path
    pos = rng.integers(0, 32, size=(bs, seq))         # position_ids path
    fc2 = np.exp(1j * np.outer(pos.ravel(), theta).reshape(bs, seq, -1))
    for label, fc in (("default (1,seq,hd/2)", fc1), ("position_ids (B,seq,hd/2)", fc2)):
        xq = rng.standard_normal((bs, seq, heads, hd))
        diff = np.abs(apply_rotary_numpy_complex(xq, fc) - apply_rotary_numpy_real(xq, fc)).max()
        check(label, diff < 1e-12, f"max diff {diff:.2e}")

    print("=" * 60)
    if FAILURES:
        print(f"❌ {len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("✅ all CPU checks passed")


if __name__ == "__main__":
    main()
