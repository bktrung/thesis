"""Pseudo-perplexity vs sequence length — the NeoBERT-Figure-2 before/after SCATTER.

Reproduces the paper's context-extension evidence in its exact visual form: each sampled long
Vietnamese-Wikipedia sequence is ONE point at (its length, its pseudo-perplexity), for two
checkpoints — before = milestone_5000000_decay (seq-1024 stage) and after = milestone_5000000_ctx4096
(seq-4096 stage) — drawn as two side-by-side scatter panels (shared y). The un-extended model's cloud
fans upward past ~2.5k; the extended model's cloud stays flat to 4096.

PPPL formula (Salazar et al. 2020): mask token i in isolation, read its CE loss l_i, and
PPPL = exp(mean_i l_i). The paper masks every position; we subsample a fixed number per sequence
(each point is still one sequence — subsampling only adds mild per-point noise, not bias). Both
checkpoints share the SALT tokenizer and the SAME (sequence, length) pairs, so points are paired and
directly comparable.

Models load via ``salt3_common.load_model_safe(..., AutoModelForMaskedLM)`` (patched remote model.py —
pure-torch SwiGLU + real-valued rotary); do NOT raw-instantiate NeoBERT.
"""
from __future__ import annotations

import contextlib
import math
import random
from pathlib import Path

# Match NeoBERT Figure 2's colours: 1024/before = blue, 4096/after = orange (CVD-validated pair).
C_BEFORE, C_AFTER = "#0072B2", "#E69F00"
WIKI_REPO, WIKI_CONFIG = "wikimedia/wikipedia", "20231101.vi"


# ── Pure helpers (torch-free, CPU-testable) ──────────────────────────────────
def sample_mask_positions(special_mask, n_masks: int, seed: int) -> list[int]:
    """Pick up to ``n_masks`` maskable positions (``special_mask[i]`` False) uniformly without
    replacement; deterministic in ``seed``. If fewer than ``n_masks`` are maskable, return all of
    them (recorded as the true count). Returns sorted indices."""
    candidates = [i for i, sp in enumerate(special_mask) if not sp]
    rng = random.Random(seed)
    if len(candidates) > n_masks:
        candidates = rng.sample(candidates, n_masks)
    return sorted(candidates)


def pppl_from_ce(ce_values) -> float:
    """Pseudo-perplexity = exp(mean cross-entropy over the independently masked positions)."""
    ce = list(ce_values)
    if not ce:
        return float("nan")
    return math.exp(sum(ce) / len(ce))


# ── Data ─────────────────────────────────────────────────────────────────────
def load_long_wiki(tokenizer, n_docs: int = 600, min_tokens: int = 4096, seed: int = 42) -> list[list[int]]:
    """Stream VN Wikipedia, tokenize (no special tokens), keep the first ``n_docs`` documents with
    >= ``min_tokens`` tokens so each can be truncated to any target length up to ``min_tokens``."""
    from datasets import load_dataset

    ds = load_dataset(WIKI_REPO, WIKI_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    docs: list[list[int]] = []
    for row in ds:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= min_tokens:
            docs.append(ids)
            if len(docs) >= n_docs:
                break
    print(f"  loaded {len(docs)} long VN-Wikipedia docs (>= {min_tokens} tokens)")
    return docs


# ── PPPL (torch) ─────────────────────────────────────────────────────────────
def pppl_at_length(model, ids, L: int, *, mask_id: int, special_ids: set, n_masks: int = 64,
                   seed: int = 42, device: str = "cuda", max_batch: int = 64,
                   mem_budget_l2: int = 4 * 4096 * 4096) -> tuple[float, int]:
    """Subsampled PPPL for one sequence truncated to length ``L``: mask each sampled position in a
    SEPARATE copy, forward the batch, read the CE at that position, return ``(exp(mean CE), n)``.

    NeoBERT's SDPA attention (xformers removed) requires an attention mask and expands it to
    [B, heads, L, L], so peak memory is O(batch * L^2). We pass an all-ones BOOL mask (cheap to
    expand, semantically full attention for these dense sequences) and shrink the batch as L grows
    (``mem_budget_l2`` caps batch * L^2) so long sequences stay within an L4's memory."""
    import torch

    trunc = list(ids[:L])
    special_mask = [tok in special_ids for tok in trunc]
    positions = sample_mask_positions(special_mask, n_masks, seed)
    if not positions:
        return float("nan"), 0

    bs = max(1, min(max_batch, mem_budget_l2 // (L * L)))   # bound batch*L^2 -> ~constant peak memory
    use_amp = torch.cuda.is_available()
    ce: list[float] = []
    base = torch.tensor(trunc, dtype=torch.long, device=device)
    ones = torch.ones(len(trunc), dtype=torch.bool, device=device)  # dense -> attend everywhere
    for start in range(0, len(positions), bs):
        chunk = positions[start:start + bs]
        inp = base.unsqueeze(0).repeat(len(chunk), 1).clone()
        for r, pos in enumerate(chunk):
            inp[r, pos] = mask_id
        with torch.no_grad():
            amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if use_amp else contextlib.nullcontext())  # bf16 matmuls -> ~2x faster, half memory
            with amp:
                out = model(input_ids=inp, attention_mask=ones.unsqueeze(0).expand(len(chunk), -1))
                logits = (out.logits if hasattr(out, "logits") else out[0]).float()
            for r, pos in enumerate(chunk):
                lp = torch.log_softmax(logits[r, pos], dim=-1)
                ce.append(float(-lp[trunc[pos]]))
    return pppl_from_ce(ce), len(positions)


def _vram_mem_budget(free_bytes: int, heads: int) -> int:
    """Pick ``mem_budget_l2`` (a cap on batch*L^2) from free VRAM so a bigger GPU uses bigger batches.
    Peak per (batch, L) is ~5*heads*batch*L^2 bytes (bool mask + bf16 scores + softmax); target ~45%
    of free memory. Floored at 4*4096^2 (batch 4 @ 4096) so an L4 still makes progress."""
    return max(4 * 4096 * 4096, int(0.45 * free_bytes / (5 * max(1, heads))))


# ── Figure ───────────────────────────────────────────────────────────────────
def _scatter_panel(ax, pts, color, title, ymax):
    if pts:
        ax.scatter([p["L"] for p in pts], [p["pppl"] for p in pts], s=10, alpha=0.5, c=color,
                   linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sequence Length")
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_pppl_scatter(before_pts, after_pts, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    ys = [p["pppl"] for p in before_pts + after_pts]
    # cap the y-axis near the bulk (like the paper's 0-20) so a few before-cloud outliers don't
    # squash the after cloud; clipped points are noted in the caption.
    ymax = float(np.percentile(ys, 98) * 1.2) if ys else 20.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _scatter_panel(axes[0], before_pts, C_BEFORE, "Before — decay (seq-1024)", ymax)
    _scatter_panel(axes[1], after_pts, C_AFTER, "After — ctx4096 (seq-4096)", ymax)
    axes[0].set_ylabel("Pseudo-Perplexity")
    fig.suptitle("Pseudo-perplexity in function of sequence length — before/after 4k context extension\n"
                 "(one point = one long VN-Wikipedia sequence; subsampled masking; shared SALT tokenizer)",
                 fontsize=11)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  figure saved -> {save_path}")
    return fig


def run_pppl_figure(before_dir, after_dir, save_path, jsonl_path, *, tokenizer, docs=None,
                    n_docs: int = 300, min_len: int = 1000, max_len: int = 4096, n_masks: int = 32,
                    seed: int = 42, device: str = "cuda") -> dict:
    """Compute one PPPL point per sequence on the SAME (sequence, length) pairs for both checkpoints,
    then draw the two-panel before/after scatter.

    CRASH-SAFE + RESUMABLE: each point is appended to ``jsonl_path`` the moment it is computed, and a
    re-run skips any (checkpoint, doc-index) already present — so a Colab drop never loses progress
    (just re-run). To start fresh (e.g. after changing n_docs/n_masks), delete the JSONL first.
    bf16 autocast + a VRAM-scaled batch keep it fast and OOM-safe on any GPU."""
    import torch
    from transformers import AutoModelForMaskedLM

    from salt3_common import append_jsonl, load_model_safe, read_jsonl

    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    if mask_id is None or (isinstance(mask_id, int) and mask_id < 0):
        raise ValueError("tokenizer has no [MASK] token; cannot compute PPPL")
    special_ids = set(tokenizer.all_special_ids)

    if docs is None:
        docs = load_long_wiki(tokenizer, n_docs=n_docs, min_tokens=max_len, seed=seed)
    if not docs:
        raise RuntimeError(f"no VN-Wikipedia docs >= {max_len} tokens; check {WIKI_REPO}/{WIKI_CONFIG}")
    # one target length per sequence, spread across [min_len, max_len]; SAME pairs for both models
    rng = random.Random(seed)
    lengths = [rng.randint(min_len, max_len) for _ in docs]

    jsonl_path = Path(jsonl_path)
    done: dict[str, set] = {}
    if jsonl_path.exists():
        for rec in read_jsonl(jsonl_path):
            done.setdefault(rec["checkpoint"], set()).add(rec.get("i"))
        print(f"  resume: {sum(len(v) for v in done.values())} points already in {jsonl_path.name}")

    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 8_000_000_000
    for label, ckpt in (("before", before_dir), ("after", after_dir)):
        todo = [i for i in range(len(docs)) if i not in done.get(label, set())]
        if not todo:
            print(f"[pppl] {label}: all {len(docs)} points already done, skipping")
            continue
        print(f"[pppl] {label}: {ckpt}  ({len(todo)}/{len(docs)} to compute)")
        model = load_model_safe(ckpt, model_cls=AutoModelForMaskedLM, device=device).eval()
        mb = _vram_mem_budget(free, getattr(model.config, "num_attention_heads", 12))
        for k, i in enumerate(todo):
            p, n = pppl_at_length(model, docs[i], lengths[i], mask_id=mask_id, special_ids=special_ids,
                                  n_masks=n_masks, seed=seed + i, device=device, mem_budget_l2=mb)
            if math.isfinite(p):
                append_jsonl(jsonl_path, {"checkpoint": label, "i": int(i), "L": int(lengths[i]),
                                          "pppl": float(p), "n": int(n)})
            if (k + 1) % 50 == 0:
                print(f"    {label}: {k + 1}/{len(todo)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {"before": [], "after": []}
    for rec in read_jsonl(jsonl_path):
        if rec["checkpoint"] in results:
            results[rec["checkpoint"]].append(rec)
    print(f"  per-point PPPL -> {jsonl_path}  (before={len(results['before'])}, after={len(results['after'])})")
    plot_pppl_scatter(results["before"], results["after"], save_path)
    return results
