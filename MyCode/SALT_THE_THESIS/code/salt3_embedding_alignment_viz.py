"""Diagram: ViDeBERTa-pruned donor rows vs NeoBERT embeddings, before/after SALT.

"Before" = raw pruned ViDeBERTa embedding rows (donor space; same 768 dims as
NeoBERT) plotted against NeoBERT's ORIGINAL input embeddings -> two disjoint
clouds (different mean / scale / orientation).
"After"  = the SALT-projected rows actually injected into the init artifact,
against the same NeoBERT embeddings -> overlapping clouds.

Both panels share ONE joint PCA basis (fit on all three matrices together) so
positions are directly comparable across panels. Anchor rows (verbatim NeoBERT
copies) and special tokens are EXCLUDED from the plotted target rows, so the
"after" overlap demonstrates the SALT local maps, not the trivially-copied
anchors. Runs on CPU; only reads weight files, never instantiates a model
(sidesteps the NeoBERT remote-code / rotary fragility entirely).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from salt3_decoder_variants import read_anchor_map, rebuild_vocab_maps

NEOBERT_REPO = "chandar-lab/NeoBERT"
VIDEBERTA_REPO = "Fsoft-AIC/videberta-base"
# Categorical palette, CVD-validated (protan dE ~110): NeoBERT blue, donor orange.
C_NEO, C_VI = "#0072B2", "#E69F00"


def _hub_state_dict(repo_id: str) -> dict[str, torch.Tensor]:
    """Raw weights from the hub, safetensors preferred, torch.bin fallback."""
    from huggingface_hub import hf_hub_download

    try:
        from safetensors.torch import load_file
        return load_file(hf_hub_download(repo_id, "model.safetensors"))
    except Exception:
        path = hf_hub_download(repo_id, "pytorch_model.bin")
        return torch.load(path, map_location="cpu", weights_only=True)


def _find_key(state: dict, suffixes: tuple[str, ...]) -> str:
    for suf in suffixes:
        hits = [k for k in state if k == suf or k.endswith(suf)]
        if hits:
            return hits[0]
    raise KeyError(f"no key ending in {suffixes}; keys start with {sorted(state)[:12]}")


def load_neobert_embedding() -> torch.Tensor:
    """NeoBERT original input embeddings [30522, 768] (nn.Embedding named 'encoder')."""
    state = _hub_state_dict(NEOBERT_REPO)
    return state[_find_key(state, ("model.encoder.weight", "encoder.weight"))].float()


def load_donor_full_embedding() -> torch.Tensor:
    """Full ViDeBERTa word embeddings. NOTE: stored under 'word_embeddings._weight'
    (DeBERTa-v3/GDES artifact), which is exactly why raw state-dict reading is the
    reliable path here (see salt3_videberta_load_probe)."""
    state = _hub_state_dict(VIDEBERTA_REPO)
    return state[_find_key(state, ("word_embeddings._weight", "word_embeddings.weight"))].float()


def load_salt_init_embedding(init_dir: Path) -> torch.Tensor:
    """The embedding matrix actually injected into our model by the SALT init."""
    from safetensors.torch import load_file

    st = Path(init_dir) / "model" / "model.safetensors"
    assert st.exists(), f"missing {st} — is this an init artifact dir?"
    state = load_file(str(st))
    return state[_find_key(state, ("model.encoder.weight", "encoder.weight"))].float()


def neobert_plot_ids(neo_emb: torch.Tensor) -> torch.Tensor:
    """NeoBERT rows worth plotting: drop [unused##] vocab slots (never trained,
    arbitrary positions) and near-zero rows (pad)."""
    from huggingface_hub import hf_hub_download

    vocab = json.loads(Path(hf_hub_download(NEOBERT_REPO, "tokenizer.json"))
                       .read_text(encoding="utf-8"))["model"]["vocab"]
    unused = {i for tok, i in vocab.items() if tok.startswith("[unused")}
    keep = [i for i in range(neo_emb.shape[0])
            if i not in unused and float(neo_emb[i].norm()) > 1e-6]
    return torch.tensor(keep)


def select_target_rows(init_dir: Path, donor_rows: int):
    """Target-vocab ids to plot: non-special, non-anchor, with a valid donor row.

    Returns (target_ids [N], donor_old_ids [N]) aligned index-for-index, so
    donor_full[donor_old_ids] is the BEFORE cloud and salt_emb[target_ids] is the
    AFTER cloud for the SAME tokens.
    """
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download

    init_dir = Path(init_dir)
    full_tok_json = hf_hub_download(VIDEBERTA_REPO, "tokenizer.json")
    target_vocab, new_to_old = rebuild_vocab_maps(init_dir / "model", full_tok_json)

    tok = AutoTokenizer.from_pretrained(str(init_dir / "model"))
    special_ids = set(tok.all_special_ids)
    anchor_ids = {target_vocab[t] for t in read_anchor_map(init_dir) if t in target_vocab}

    target_ids, old_ids = [], []
    for new_id, old_id in new_to_old.items():
        if new_id in special_ids or new_id in anchor_ids or old_id >= donor_rows:
            continue
        target_ids.append(new_id)
        old_ids.append(old_id)
    print(f"  target rows plotted: {len(target_ids):,} "
          f"(excluded {len(anchor_ids):,} anchors, {len(special_ids)} specials)")
    return torch.tensor(target_ids), torch.tensor(old_ids)


def joint_pca_2d(mats: list[torch.Tensor]):
    """One PCA basis fit on the concatenation, applied to each matrix."""
    X = torch.cat(mats)
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    W = evecs[:, -2:].flip(1)                       # top-2 PCs, descending
    var_frac = (evals.flip(0)[:2] / evals.sum()).tolist()
    return [(m - mu) @ W for m in mats], var_frac


def alignment_stats(neo: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> dict:
    """Quantitative companion to the figure, computed in the full 768-d space."""
    c_neo = neo.mean(0)
    return {
        "centroid_dist_before": float((before.mean(0) - c_neo).norm()),
        "centroid_dist_after": float((after.mean(0) - c_neo).norm()),
        "row_norm_neo": float(neo.norm(dim=1).mean()),
        "row_norm_before": float(before.norm(dim=1).mean()),
        "row_norm_after": float(after.norm(dim=1).mean()),
    }


def _scatter_panel(ax, neo2d, vi2d, vi_label, note, n_scatter, seed):
    g = torch.Generator().manual_seed(seed)
    for pts, color, label in ((neo2d, C_NEO, "NeoBERT (original)"), (vi2d, C_VI, vi_label)):
        idx = torch.randperm(pts.shape[0], generator=g)[:n_scatter]
        sub = pts[idx]
        ax.scatter(sub[:, 0], sub[:, 1], s=4, alpha=0.3, c=color, label=label,
                   linewidths=0, rasterized=True)
        c = pts.mean(0)                              # centroid + direct label (contrast relief)
        ax.scatter([c[0]], [c[1]], marker="X", s=120, c=color, edgecolors="white",
                   linewidths=1.2, zorder=5)
        ax.annotate(label, (c[0], c[1]), xytext=(8, 8), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#333333")
    ax.set_title(note, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_before_after(neo2d, before2d, after2d, var_frac, stats, save_path=None,
                      n_scatter=4000, seed=42):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    _scatter_panel(axes[0], neo2d, before2d, "ViDeBERTa-pruned (raw)",
                   f"Before SALT — centroid dist {stats['centroid_dist_before']:.1f}",
                   n_scatter, seed)
    _scatter_panel(axes[1], neo2d, after2d, "SALT-projected (injected)",
                   f"After SALT — centroid dist {stats['centroid_dist_after']:.1f}",
                   n_scatter, seed)
    axes[0].set_ylabel(f"PC2 ({var_frac[1]:.0%} var)")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({var_frac[0]:.0%} var)")
    fig.suptitle("Vietnamese donor embeddings vs NeoBERT space — before/after SALT projection\n"
                 "(joint PCA basis; anchors & specials excluded — overlap is the local maps, not copies)",
                 fontsize=11)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  figure saved -> {save_path}")
    return fig


def run_diagram(init_dir: str | Path, save_path: str | Path | None = None,
                n_scatter: int = 4000, seed: int = 42) -> dict:
    """Load everything, compute joint PCA + stats, draw the two-panel figure."""
    init_dir = Path(init_dir)
    print(f"Init artifact: {init_dir.name}")
    neo = load_neobert_embedding()
    donor_full = load_donor_full_embedding()
    salt = load_salt_init_embedding(init_dir)
    assert neo.shape[1] == donor_full.shape[1] == salt.shape[1], \
        f"hidden dims differ: neo {neo.shape}, donor {donor_full.shape}, salt {salt.shape}"

    target_ids, old_ids = select_target_rows(init_dir, donor_full.shape[0])
    neo_keep = neo[neobert_plot_ids(neo)]
    before, after = donor_full[old_ids], salt[target_ids]

    stats = alignment_stats(neo_keep, before, after)
    print("  " + json.dumps(stats, indent=2).replace("\n", "\n  "))
    (neo2d, before2d, after2d), var_frac = joint_pca_2d([neo_keep, before, after])
    plot_before_after(neo2d, before2d, after2d, var_frac, stats, save_path, n_scatter, seed)
    return stats
