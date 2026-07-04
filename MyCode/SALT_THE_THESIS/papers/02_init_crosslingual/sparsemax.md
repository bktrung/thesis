---
title: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
authors: Martins, Astudillo
year: 2016
venue: ICML 2016
arxiv: 1602.02068
url: https://arxiv.org/abs/1602.02068
tags: [init, math, attention]
---

# Sparsemax

## Core idea
A drop-in alternative to softmax that returns a **sparse** probability distribution — many outputs
are **exactly zero**. It is the **Euclidean projection of the logits onto the probability simplex**;
unlike softmax (which is never exactly zero), it can completely ignore irrelevant elements.

## Key math / architecture details
- Definition: `sparsemax(z) = argmin_{p ∈ Δ} ‖p − z‖²`, where `Δ` is the probability simplex.
- **Closed form (thresholding):**
  1. sort logits `z_(1) ≥ z_(2) ≥ …`;
  2. find `k(z) = max{ k : 1 + k·z_(k) > Σ_{j≤k} z_(j) }`;
  3. threshold `τ(z) = (Σ_{j≤k(z)} z_(j) − 1) / k(z)`;
  4. output `p_i = max(0, z_i − τ(z))`.
- Everything below the threshold is clipped to 0 → support is a small subset; the surviving mass
  sums to 1. Piecewise-linear and differentiable a.e. (has a defined Jacobian for backprop).
- This is exactly the implementation in `salt3_decoder_variants.py::sparsemax` (cumsum + threshold).

## Results / why it matters
Comparable accuracy to softmax on attention and multi-label tasks, with **selective, interpretable**
focus. The sparsity is the useful property here: a weighting that picks a *few* relevant items.

## How NeoBERT / SALT3 uses this
Sparsemax is the **combination rule** for SALT3's embedding init (inherited via FOCUS → [[focus]]).
For each target token, similarities to mined anchors are passed through **sparsemax**, so the new
embedding is a convex combination of only a **handful of the most relevant donor anchors** — not a
dense blur over all of them. The repo cites it verbatim ("Sparsemax, Martins & Astudillo 2016") and
uses it in two places: `sparsemax(ft @ anchor_ftᵀ)` for per-token anchor weights, and the
`sparsemax → min-norm lstsq` fallback in the per-token decoder maps.

## Relation: [[focus]] [[wechsel]] [[salt]]
