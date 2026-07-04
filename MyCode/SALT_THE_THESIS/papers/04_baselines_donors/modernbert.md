---
title: "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT)"
authors: Warner, Chaffin, Clavié, Weller, Hallström, Taghadouini, et al.
year: 2024
venue: preprint (Answer.AI / LightOn)
arxiv: 2412.13663
url: https://arxiv.org/abs/2412.13663
tags: [baseline, encoder]
---

# ModernBERT

## Core idea
The other "modernized BERT": take an encoder and apply the full modern stack — RoPE, GeGLU,
**alternating local/global attention**, unpadding/sequence-packing, FlashAttention, **8,192** context,
~2T training tokens — for a large Pareto improvement in speed, memory, and long-context quality over
BERT/RoBERTa. Direct contemporary of NeoBERT with overlapping but distinct choices.

## Key math / architecture details
- **RoPE** positions ([[rope]]); **GeGLU** FFN (the GELU-gated GLU, sibling of SwiGLU → [[swiglu]]).
- **Alternating attention:** most layers use *local* sliding-window attention, a few use *global* —
  cheaper long-context than all-global. (NeoBERT instead uses full attention with xFormers/Flash.)
- **Unpadding / sequence packing** + FlashAttention ([[flash-attention]]) for throughput.
- Sizes: **base 149M**, **large 395M**; trained on ~2T tokens of mixed web/code.
- vs **NeoBERT**: NeoBERT is 250M with a deep-narrow 28×768 ratio, **SwiGLU + Pre-RMSNorm**, full
  attention, 4,096 context, and reports beating ModernBERT on MTEB under identical fine-tuning.

## Results / why it matters
SOTA across classification and single/multi-vector retrieval (incl. code); the strongest "modern
encoder" baseline alongside NeoBERT.

## How NeoBERT / SALT3 uses this
ModernBERT is a **primary baseline NeoBERT compares against and outperforms** on MTEB — part of the
justification for choosing NeoBERT as SALT3's base. It is also a **design reference**: it shows the
same modern-component thesis (RoPE/GLU/Flash/long-context) the thesis relies on, via a slightly
different recipe, which is useful for the related-work contrast in the report. SALT3 itself does not
adapt ModernBERT, but its components (GeGLU vs SwiGLU, local/global vs full attention) are the natural
ablation neighbors. → [[neobert]].

## Relation: [[neobert]] [[bert]] [[roberta]] [[rope]] [[swiglu]] [[flash-attention]]
