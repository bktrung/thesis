---
title: "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
authors: Loshchilov, Hsieh, Sun, Ginsburg
year: 2024
venue: preprint (NVIDIA)
arxiv: 2410.01131
url: https://arxiv.org/abs/2410.01131
tags: [architecture, normalization, variant]
---

# nGPT (Normalized Transformer)

## Core idea
Constrain **every** vector (embeddings, hidden states, attention/MLP outputs, weight matrix rows)
to unit norm so all representations live on a **hypersphere**. Each layer then performs a
**normalized update**: it moves the hidden state a learnable step *toward* the sublayer's output
direction and re-normalizes, like a step of Riemannian optimization on the sphere. Removes the need
for explicit LayerNorm/RMSNorm layers and reportedly cuts training steps to convergence by 4–20×.

## Key math / architecture details
- All hidden states normalized to the unit hypersphere; dot products become cosine similarities.
- **Normalized residual update (eigen-step):** `h ← Norm(h + α·(h_sublayer − h))` with a learnable,
  per-dimension step size `α` (the "eigen learning rate"), instead of a plain residual add.
- Weight matrices are normalized along the embedding dimension; logits/attention use learnable
  scaling factors to restore range.
- No final LayerNorm — the sphere constraint plays the normalization role.

## Results / why it matters
Faster convergence and good stability; a clean geometric reframing of why normalization helps.
Relevant as the design behind NeoBERT's alternative encoder block.

## How NeoBERT / SALT3 uses this
NeoBERT ships a **`NormNeoBERT`** variant whose `NormEncoderBlock` uses these **normalized
updates** and **omits the final norm** — `NeoBERT/docs/architecture.md` explicitly calls it
"NGPT-style." It is a secondary architecture arm, not the 250M-opt model the thesis trains, but the
thesis's init work cares about **per-row norms** (the hypersphere intuition): SALT3's
`meannorm` scale-matching and the encoder/decoder norm diagnostics (`salt3_init_forensics.py`,
`14_freeze_align_encoder_decoder_init`) reason about embedding rows in exactly this normalized
geometry.

## Relation: [[rmsnorm]] [[neobert]] [[transformer]]
