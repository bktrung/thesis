---
title: Gaussian Error Linear Units (GELUs)
authors: Hendrycks, Gimpel
year: 2016
venue: preprint
arxiv: 1606.08415
url: https://arxiv.org/abs/1606.08415
tags: [architecture, activation]
---

# GELU

## Core idea
A smooth activation that weights an input by the probability it is kept under a Gaussian — a
"stochastic-regularizer-as-deterministic-nonlinearity." Behaves between ReLU and a soft gate, and
became the default activation in BERT/GPT-2 before SwiGLU.

## Key math / architecture details
- `GELU(x) = x · Φ(x)`, where `Φ` is the standard-normal CDF.
- Tanh approximation (used in practice):
  `GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])`.
- Sigmoid approximation: `x·σ(1.702x)` (≈ SiLU/Swish, linking GELU to the Swish in [[swiglu]]).
- Smooth and non-monotonic near 0 → nonzero gradient for small negative inputs, unlike ReLU.

## Results / why it matters
Outperformed ReLU/ELU across vision and NLP benchmarks; adopted by BERT, GPT-2, RoBERTa, ViT.
SwiGLU later improved on plain GELU FFNs, but GELU remains the standard *non-gated* baseline.

## How NeoBERT / SALT3 uses this
GELU is NeoBERT's **alternative MLP activation**: the `base`/non-opt config sets
`hidden_act: gelu` (and `rms_norm: false`, `rope: false`) — i.e. the "classic BERT-like" arm —
while the 250M-opt model the thesis uses runs **SwiGLU**. GELU is the comparison point that
motivates the SwiGLU choice. SALT3's donor models PhoBERT/ViDeBERTa ([[phobert]], [[videberta]])
are GELU-era BERT/DeBERTa encoders, so understanding GELU helps reason about donor-vs-NeoBERT
activation mismatch during embedding transfer.

## Relation: [[swiglu]] [[bert]] [[neobert]]
