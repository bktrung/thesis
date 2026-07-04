---
title: On Layer Normalization in the Transformer Architecture
authors: Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu
year: 2020
venue: ICML 2020
arxiv: 2002.04745
url: https://arxiv.org/abs/2002.04745
tags: [architecture, normalization, optimization]
---

# Pre-LN vs Post-LN

## Core idea
*Where* you put the normalization changes the gradient behavior at initialization. The original
Transformer is **Post-LN** (`LayerNorm(x + Sublayer(x))`); this paper shows **Pre-LN**
(`x + Sublayer(LayerNorm(x))`) has much better-behaved gradients, removing the need for a carefully
tuned learning-rate warmup and making deep models trainable.

## Key math / architecture details
- **Post-LN:** gradients near the output layer scale with depth; training is unstable without a
  warmup stage, and is sensitive to the warmup length and peak LR.
- **Pre-LN:** the residual path is identity (norm is *inside* the residual branch), so expected
  gradients are well-scaled and roughly **depth-independent** at init. Warmup can be reduced or
  removed; larger LRs are usable.
- Trade-off: Pre-LN can slightly underperform a *perfectly tuned* Post-LN, but is far more robust —
  the right choice for deep stacks (NeoBERT is 28 layers).
- Pre-norm is what makes a clean **identity residual highway** through the network, important for
  both stable deep training and for continued pre-training.

## Results / why it matters
Pre-LN Transformers train stably without warmup and converge faster in wall-clock on MT and BERT
pre-training. Combined with RMSNorm, "Pre-RMSNorm" is the modern default.

## How NeoBERT / SALT3 uses this
NeoBERT uses **pre-norm** placement with RMSNorm ([[rmsnorm]]) — see
`NeoBERT/docs/architecture.md` ("pre-norm attn + MLP"). For SALT3 this is load-bearing: a freshly
re-initialized embedding enters the network through a Pre-RMSNorm, and the identity residual
highway is *why* continued pre-training can repair a partially-mismatched init without destabilizing
the deep body. It also explains the thesis's **warmup** choices in CPT → [[warmup]].

## Relation: [[transformer]] [[rmsnorm]] [[neobert]] [[warmup]]
