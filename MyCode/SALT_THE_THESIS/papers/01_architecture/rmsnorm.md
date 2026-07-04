---
title: Root Mean Square Layer Normalization
authors: Zhang, Sennrich
year: 2019
venue: NeurIPS 2019
arxiv: 1910.07467
url: https://arxiv.org/abs/1910.07467
tags: [architecture, normalization]
---

# RMSNorm

## Core idea
LayerNorm does two things: **re-centering** (subtract the mean) and **re-scaling** (divide by
std). RMSNorm hypothesizes that re-centering is **dispensable** and keeps only re-scaling, using
the root-mean-square of the activations. Result: same quality, less compute, no bias/mean to track.

## Key math / architecture details
- `RMSNorm(x) = (x / RMS(x)) ⊙ γ`, with `RMS(x) = sqrt( (1/d) Σ_i x_i² + ε )`.
- vs `LayerNorm(x) = ((x − μ)/sqrt(σ² + ε)) ⊙ γ + β`. RMSNorm drops `μ`, `σ`, and the bias `β`.
- Keeps **re-scaling invariance** and the implicit learning-rate-adaptation property of LayerNorm;
  gives up re-centering invariance (shown empirically unnecessary).
- 7%–64% wall-clock speedup on the norm op depending on architecture.
- **Placement matters:** with **pre-norm** wiring (`x + Sublayer(RMSNorm(x))`) it gives stable
  deep-network training → [[pre-ln]].

## Results / why it matters
Comparable to LayerNorm across MT, language modeling, and CNN/RNN tasks at lower cost. Now the
default norm in LLaMA-family models and modern encoders.

## How NeoBERT / SALT3 uses this
NeoBERT sets `rms_norm: true`, `norm_eps: 1e-5`: **Pre-RMSNorm** before attention and FFN in every
block, plus a final RMSNorm after the stack (the `NormNeoBERT`/nGPT variant omits the final norm →
[[ngpt]]). Implemented in `NeoBERT/src/neobert/model/rmsnorm.py`. SALT3 inherits this unchanged; it
matters for the thesis because a mis-scaled re-initialized embedding row gets renormalized by the
first RMSNorm, so the init's **per-row norm** (the `meannorm` scale-matching in
`salt3_decoder_variants.py`) interacts directly with this layer.

## Relation: [[transformer]] [[pre-ln]] [[neobert]] [[ngpt]]
