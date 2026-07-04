"""SALT3 projection diagnostics.

Tools to answer one question: is the SALT embedding projection (ViDeBERTa -> NeoBERT)
actually landing tokens in NeoBERT's geometry, or is the local-linear math the leak?

The projection in 01_init_embeddings_salt3.ipynb (cell 17) is Locally-Linear
Embedding: pick anchors by FastText cosine, reconstruct the token as a linear
combo of those anchors in ViDeBERTa space, reuse the combo weights on the
anchors' NeoBERT embeddings. These functions measure how well that holds up.

All functions take the in-notebook tensors directly so they can be called right
after cell 17 without reloading anything. Tensors are expected on the same device;
CPU is fine and used for the heavier sweeps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo 2016) — must mirror the notebook's copy."""
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cum_sums = torch.cumsum(sorted_logits, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype).expand_as(logits)
    k_z = (1 + k * sorted_logits > cum_sums).sum(dim=dim, keepdim=True).float()
    tau = (torch.gather(cum_sums, dim, (k_z - 1).long()) - 1) / k_z
    return torch.clamp(logits - tau, min=0.0)


def _reconstruct(
    ft_vec: torch.Tensor,            # [768] normalized FastText vec of the query token
    target_emb_vec: torch.Tensor,    # [768] ViDeBERTa embedding of the query token
    anchor_ft: torch.Tensor,         # [A,768] normalized FastText vecs of anchor pool
    anchor_Et: torch.Tensor,         # [A,768] ViDeBERTa embeddings of anchor pool
    anchor_Es: torch.Tensor,         # [A,768] NeoBERT embeddings of anchor pool
    min_anchors: int,
    ridge: float,
    exclude_idx: int | None = None,
):
    """Mirror cell-17 projection for one token; return (vec, k, max_abs_coef, method).

    ridge>0 regularizes the anchor Gram (the missing piece vs bare pinv): coefficients
    c = (G + ridge*I)^-1 E_t t, new = E_s^T c. max_abs_coef flags extrapolation
    (|c| >> 1 means the token is reconstructed by large cancelling anchor weights —
    the instability that corrupts direction).
    """
    sim = anchor_ft @ ft_vec
    if exclude_idx is not None:
        sim = sim.clone()
        sim[exclude_idx] = -1e9  # leave-one-out: hide the token's own anchor row
    support = _sparsemax(sim, dim=0)
    nz = torch.nonzero(support).squeeze(1)
    if len(nz) == 0:
        return None, 0, float("nan"), "fallback"

    k = len(nz)
    Et = anchor_Et[nz]               # [k,768]
    Es = anchor_Es[nz]               # [k,768]
    if k >= min_anchors:
        G = Et @ Et.T                # [k,k] anchor Gram in ViDeBERTa space
        G = G + ridge * torch.eye(k, device=G.device, dtype=G.dtype)
        c = torch.linalg.solve(G, Et @ target_emb_vec)   # LLE coords
        vec = Es.T @ c
        return vec, k, float(c.abs().max()), "lstsq"
    w = support[nz].unsqueeze(1)
    vec = (Es * w).sum(0) / w.sum()
    return vec, k, float(w.max()), "wavg"


def leave_one_out_anchor_error(
    anchor_map: dict,
    source_vocab: dict,
    target_vocab: dict,
    new_to_old: dict,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    ft,
    min_anchors: int = 8,
    ridge: float = 1e-3,
    max_anchors: int | None = None,
    selection: str = "fasttext",
    quiet: bool = False,
) -> pd.DataFrame:
    """Ground-truth test of the projection math.

    For each anchor: hide it, project it from the OTHER anchors with the real
    pipeline, compare to its TRUE NeoBERT embedding. High cosine => the SALT math
    reconstructs known points => the regression is elsewhere. Low cosine => the
    projection itself is the leak.

    selection: which space picks the sparsemax anchor support.
      'fasttext' — the current pipeline (cross-lingual bridge).
      'donor'    — SALT-paper-faithful: similarity in the donor PLM's own space.
    The reconstruction coefficients are always fit in donor space; if the two
    selections diverge a lot, the FastText bridge is the deviation to blame.

    Returns per-anchor DataFrame and prints the summary (unless quiet).
    """
    vi_tokens = list(anchor_map.keys())
    if max_anchors:
        vi_tokens = vi_tokens[:max_anchors]

    def ftv(tok):
        v = torch.tensor(ft.get_word_vector(tok.replace("▁", "")), dtype=torch.float32)
        return F.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)

    anchor_Et = torch.stack([target_embeddings[new_to_old[target_vocab[t]]] for t in vi_tokens]).float()
    anchor_Es = torch.stack([source_embeddings[source_vocab[anchor_map[t]]] for t in vi_tokens]).float()
    if selection == "donor":
        anchor_sel = F.normalize(anchor_Et, p=2, dim=1)
    else:
        anchor_sel = torch.stack([ftv(t) for t in vi_tokens])

    rows = []
    for i, tok in enumerate(vi_tokens):
        pred, k, maxc, method = _reconstruct(
            anchor_sel[i], anchor_Et[i], anchor_sel, anchor_Et, anchor_Es,
            min_anchors, ridge, exclude_idx=i,
        )
        truth = anchor_Es[i]
        cos = float(F.cosine_similarity(pred.unsqueeze(0), truth.unsqueeze(0))) if pred is not None else float("nan")
        norm_ratio = float(pred.norm() / truth.norm().clamp(min=1e-8)) if pred is not None else float("nan")
        rows.append(dict(token=tok, k=k, method=method, cos=cos, norm_ratio=norm_ratio, max_coef=maxc))

    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["cos"])
    if quiet:
        return df
    print(f"── Leave-one-out anchor reconstruction (selection={selection}) ──")
    print(f"  anchors tested        : {len(ok)}")
    print(f"  cosine  mean/median   : {ok.cos.mean():.3f} / {ok.cos.median():.3f}")
    print(f"  cosine  >0.7 fraction : {(ok.cos > 0.7).mean():.1%}   (>0.5: {(ok.cos > 0.5).mean():.1%})")
    print(f"  norm ratio mean       : {ok.norm_ratio.mean():.3f}  (1.0 = perfect scale)")
    print(f"  |coef| max  p50/p95   : {ok.max_coef.median():.2f} / {ok.max_coef.quantile(0.95):.2f}  (>>1 = extrapolation)")
    print(f"  verdict               : "
          f"{'MATH OK — look elsewhere' if ok.cos.median() > 0.7 else 'PROJECTION IS THE LEAK' if ok.cos.median() < 0.5 else 'BORDERLINE'}")
    return df


def _knn_overlap(A: torch.Tensor, B: torch.Tensor, k: int) -> float:
    """Mean top-k neighbor overlap between two row-aligned, L2-normalized matrices.
    Chance level ≈ k/(n−1)."""
    overlaps = []
    for r in range(A.shape[0]):
        na = (A @ A[r]).topk(k + 1).indices[1:]
        nb = (B @ B[r]).topk(k + 1).indices[1:]
        overlaps.append(len(set(na.tolist()) & set(nb.tolist())) / k)
    return float(np.mean(overlaps))


def knn_preservation(
    target_embeddings: torch.Tensor,
    new_embeddings: torch.Tensor,
    new_to_old: dict,
    token_ids: list[int],
    k: int = 10,
) -> float:
    """Neighborhood overlap: do a token's ViDeBERTa neighbors stay neighbors after
    projection into NeoBERT space? Sampled over token_ids (new-vocab ids). 1.0 = the
    projection preserves local semantic structure; ~0 = it scrambles it."""
    old_ids = [new_to_old[i] for i in token_ids]
    Et = F.normalize(target_embeddings[old_ids].float(), dim=1)
    Es = F.normalize(new_embeddings[token_ids].float(), dim=1)
    val = _knn_overlap(Et, Es, k)
    print(f"── kNN preservation (k={k}, n={len(token_ids)}): {val:.3f} "
          f"({'good' if val > 0.4 else 'weak' if val > 0.2 else 'scrambled'}) ──")
    return val


def donor_space_soundness(donor_emb: torch.Tensor, donor_vocab: dict, ft, n: int = 1000, k: int = 10,
                          seed: int = 0) -> dict:
    """Rung 1: does the DONOR's own embedding geometry encode Vietnamese semantics?

    No projection involved — this isolates the donor itself (DeBERTa-v3/RTD risk:
    embeddings trained for replaced-token detection need not be semantic). Samples
    full-word Vietnamese tokens from the donor vocab and compares donor-embedding
    kNN against FastText-vi kNN over the same words.

    Two FastText variants are reported because multi-syllable `_`-joined words
    (PhoBERT style) are OOV in cc.vi.300 and would bias the score down:
      raw      — ft vector of the surface as-is (subword backoff when OOV)
      syllable — mean of per-syllable vectors (split on '_')
    Only words whose every syllable is in the FastText vocab are sampled, so the
    reference geometry is real, not subword-guessed.

    Gates: best variant >0.2 → donor usable; <0.05 ≈ chance → donor is the
    problem (prioritize the other donor arm).
    """
    rng = np.random.default_rng(seed)
    toks = list(donor_vocab.keys())
    has_sp = any(t.startswith("▁") for t in toks[:50000])  # sentencepiece (ViDeBERTa) vs fastBPE @@ (PhoBERT)
    cands = []
    for t in toks:
        if has_sp:
            if not t.startswith("▁"):
                continue  # continuation piece, not a full word
            surface = t[1:]
        else:
            if t.endswith("@@"):
                continue  # fastBPE continuation marker
            surface = t
        if len(surface) < 2 or not surface.replace("_", "").isalpha():
            continue
        syls = surface.split("_")
        if any(ft.get_word_id(s) < 0 for s in syls):
            continue
        cands.append((t, surface, syls))
    if len(cands) < 2 * k:
        raise ValueError(f"only {len(cands)} usable Vietnamese full-word tokens in donor vocab — wrong vocab/ft?")
    sel = [cands[i] for i in rng.choice(len(cands), size=min(n, len(cands)), replace=False)]

    D = torch.stack([donor_emb[donor_vocab[t]] for t, _, _ in sel]).float()
    E = F.normalize(D, dim=1)
    # Mean-centered variant: anisotropic embedding tables (all rows in a narrow cone)
    # can flatten raw-cosine kNN even when relative structure exists — centering
    # removes the common component so the verdict is not a cone artifact.
    E_c = F.normalize(D - D.mean(0), dim=1)
    raw = torch.stack([torch.tensor(ft.get_word_vector(s)) for _, s, _ in sel]).float()
    syl = torch.stack([torch.stack([torch.tensor(ft.get_word_vector(x)) for x in ss]).mean(0)
                       for _, _, ss in sel]).float()
    Fr, Fs = F.normalize(raw, dim=1), F.normalize(syl, dim=1)
    out = dict(n=len(sel), k=k,
               overlap_raw=_knn_overlap(E, Fr, k),
               overlap_syllable=_knn_overlap(E, Fs, k),
               overlap_raw_centered=_knn_overlap(E_c, Fr, k),
               overlap_syllable_centered=_knn_overlap(E_c, Fs, k))
    best = max(out["overlap_raw"], out["overlap_syllable"],
               out["overlap_raw_centered"], out["overlap_syllable_centered"])
    out["verdict"] = ("DONOR SOUND — geometry carries VI semantics" if best > 0.2
                      else "DONOR BROKEN — ~chance vs FastText; swap donor" if best < 0.05
                      else "WEAK — some signal; projection must be near-lossless to survive")
    print(f"── Donor-space soundness (n={out['n']}, k={k}, chance≈{k/(out['n']-1):.3f}) ──")
    print(f"  kNN overlap vs FastText: raw {out['overlap_raw']:.3f} | syllable-mean {out['overlap_syllable']:.3f}")
    print(f"  mean-centered          : raw {out['overlap_raw_centered']:.3f} | syllable-mean {out['overlap_syllable_centered']:.3f}")
    print(f"  verdict (best variant) : {out['verdict']}   (gates: >0.2 usable, <0.05 broken)")
    return out


def norm_audit(source_embeddings, target_embeddings, new_embeddings) -> dict:
    """Row-norm statistics across the three spaces (NeoBERT / ViDeBERTa / projected)."""
    def stat(t):
        n = t.float().norm(dim=1)
        n = n[n > 1e-6]
        return dict(mean=float(n.mean()), std=float(n.std()), p5=float(n.quantile(.05)), p95=float(n.quantile(.95)))
    out = {"neobert": stat(source_embeddings), "videberta": stat(target_embeddings), "projected": stat(new_embeddings)}
    print("── Norm audit (row L2) ──")
    for k, v in out.items():
        print(f"  {k:10s}: mean {v['mean']:.3f}  std {v['std']:.3f}  p5/p95 {v['p5']:.3f}/{v['p95']:.3f}")
    return out


def visualize_spaces(
    source_embeddings,
    new_embeddings,
    anchor_neo_ids: list[int],
    sample_new_ids: list[int],
    loo_df: pd.DataFrame | None = None,
    save_path: str | None = None,
):
    """PCA(NeoBERT space) scatter: NeoBERT anchors, NeoBERT cloud, and projected
    non-anchor tokens — so you can see whether projected tokens fall inside the
    NeoBERT manifold or drift to its own clump. Also a norm histogram.

    PCA basis is fit on NeoBERT embeddings (the target frame). ViDeBERTa-before is
    a different basis, so it is NOT overlaid here — neighborhood preservation
    (knn_preservation) is the right cross-space metric, not a shared scatter.
    """
    import matplotlib.pyplot as plt

    S = source_embeddings.float()
    mu = S.mean(0)
    _, _, V = torch.pca_lowrank(S - mu, q=2)

    def proj(t):
        return ((t.float() - mu) @ V).cpu().numpy()

    neo_bg = proj(S[torch.randint(0, S.shape[0], (2000,))])
    neo_anc = proj(S[torch.tensor(anchor_neo_ids)])
    proj_tok = proj(new_embeddings[torch.tensor(sample_new_ids)])

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].scatter(neo_bg[:, 0], neo_bg[:, 1], s=3, c="lightgray", label="NeoBERT cloud")
    ax[0].scatter(neo_anc[:, 0], neo_anc[:, 1], s=14, c="tab:blue", label="NeoBERT anchors")
    ax[0].scatter(proj_tok[:, 0], proj_tok[:, 1], s=14, c="tab:red", alpha=.6, label="projected VI tokens")
    ax[0].set_title("PCA in NeoBERT space — do projected tokens land in the manifold?")
    ax[0].legend()

    if loo_df is not None:
        ok = loo_df.dropna(subset=["cos"])
        ax[1].hist(ok.cos, bins=40, color="tab:purple")
        ax[1].axvline(ok.cos.median(), c="k", ls="--", label=f"median {ok.cos.median():.2f}")
        ax[1].set_title("Leave-one-out anchor reconstruction cosine")
        ax[1].set_xlabel("cosine(pred, true NeoBERT)")
        ax[1].legend()
    else:
        sn = S.norm(dim=1).cpu().numpy()
        pn = new_embeddings[torch.tensor(sample_new_ids)].float().norm(dim=1).cpu().numpy()
        ax[1].hist(sn, bins=40, alpha=.5, label="NeoBERT", density=True)
        ax[1].hist(pn, bins=40, alpha=.5, label="projected", density=True)
        ax[1].set_title("Row-norm distribution")
        ax[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"saved -> {save_path}")
    plt.show()
