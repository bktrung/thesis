---
title: GLU Variants Improve Transformer
authors: Shazeer
year: 2020
venue: preprint
arxiv: 2002.05202
url: https://arxiv.org/abs/2002.05202
tags: [architecture, activation, ffn]
---

# SwiGLU (GLU Variants Improve Transformer)

## Core idea
Replace the Transformer FFN's single ReLU/GELU projection with a **Gated Linear Unit**: two
parallel linear projections of the input, one passed through an activation and used to **gate**
(elementwise-multiply) the other. The Swish-gated variant, **SwiGLU**, consistently gives the
best quality among the tested gates.

## Key math / architecture details
- Baseline FFN: `FFN(x) = act(xW₁)W₂`.
- GLU family (three matrices `W, V, W₂`):
  - `GLU(x)   = (σ(xW) ⊗ xV)W₂`
  - `GEGLU(x) = (GELU(xW) ⊗ xV)W₂`
  - **`SwiGLU(x) = (Swish_β(xW) ⊗ xV)W₂`**, with `Swish_β(z)=z·σ(βz)` (β=1 ≡ SiLU).
  - `Bilinear(x) = (xW ⊗ xV)W₂`
  (`⊗` = elementwise product.)
- **Parameter-matching:** GLU FFNs use **three** weight matrices instead of two. To hold the
  parameter/FLOP budget fixed, the hidden width is scaled by **2/3** (e.g. `4d → 8d/3`).
  NeoBERT rounds this to a **multiple of 8** for hardware alignment.
- Cost: one extra matmul vs vanilla FFN, but better quality per parameter.

## Results / why it matters
On a T5 span-corruption setup, GEGLU/SwiGLU FFNs lowered perplexity and improved GLUE/SuperGLUE
over ReLU and GELU FFNs. Shazeer's closing line ("We offer no explanation… divine benevolence")
is famous; empirically it just works, and it is now standard (LLaMA, PaLM, NeoBERT).

## How NeoBERT / SALT3 uses this
NeoBERT sets `hidden_act: swiglu` with the 2/3-width, multiple-of-8 rule — the FFN in every
encoder block is SwiGLU, as documented in `NeoBERT/docs/architecture.md`. The GELU FFN ([[gelu]])
remains available as the alternative (`base` config uses `hidden_act: gelu`). SALT3 trains through
these SwiGLU FFNs unchanged; only the embedding/decoder endpoints are re-initialized.

## Relation: [[transformer]] [[gelu]] [[neobert]] [[llama]]
