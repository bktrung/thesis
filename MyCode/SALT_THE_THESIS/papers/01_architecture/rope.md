---
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: Su, Lu, Pan, Murtadha, Wen, Liu
year: 2021
venue: preprint (later Neurocomputing 2024)
arxiv: 2104.09864
url: https://arxiv.org/abs/2104.09864
tags: [architecture, positional]
---

# RoPE (Rotary Position Embedding)

## Core idea
Inject position by **rotating** the query and key vectors by an angle proportional to their
absolute position, *before* the dot product. Because a dot product of two rotated vectors
depends only on the *difference* of their rotation angles, attention scores become a function of
**relative** position (m − n) while the mechanism itself only ever uses absolute positions. No
added position vectors, no extra parameters.

## Key math / architecture details
- Split each head's `d`-dim vector into `d/2` 2-D pairs. Pair `i` is rotated by angle `m·θ_i` at
  position `m`, with frequencies
  `θ_i = 10000^(−2i/d)` (geometric, like sinusoidal PE).
- For a query at position `m`: `q̃_m = R_m q_m`, key at `n`: `k̃_n = R_n k_n`, where `R_m` is a
  block-diagonal rotation matrix. Then
  `⟨q̃_m, k̃_n⟩ = q_mᵀ R_{n−m} k_n` — depends only on `n − m`.
- **Long-range decay:** high-frequency pairs decorrelate quickly with distance, so attention
  logits have a built-in tendency to decay as |m − n| grows.
- Implemented efficiently as elementwise multiply with precomputed `cos(mθ)`/`sin(mθ)` tables and
  a "rotate-half" operation; applied **per block to Q and K only** (not V).
- The **base** (10000) controls the wavelength range and is the knob for context-length extension.

## Results / why it matters
Matches or beats learned/relative position schemes, extrapolates better to longer sequences, and
is compatible with linear/efficient attention. Now the default in LLaMA, GPT-NeoX, ModernBERT,
NeoBERT, etc.

## How NeoBERT / SALT3 uses this
NeoBERT applies RoPE to Q/K inside every encoder block (`rope: true`), replacing BERT's learned
absolute position embeddings — this is what enables the **4,096-token** context. The repo's
`NeoBERT/src/neobert/model/rotary.py` implements the rotation; the thesis notebooks
`09/11/12_*` debug a **rotary complex-NaN** issue in this path. SALT3 leaves RoPE untouched:
re-initializing the vocabulary changes nothing about positions, so all length behavior carries
over to the Vietnamese model for free.

## Relation: [[transformer]] [[neobert]] [[flash-attention]]
