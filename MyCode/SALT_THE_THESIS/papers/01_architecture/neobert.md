---
title: "NeoBERT: A Next-Generation BERT"
authors: Le Breton, Fournier, El Mezouar, Morris, Chandar
year: 2025
venue: preprint (Chandar Lab)
arxiv: 2502.19587
url: https://arxiv.org/abs/2502.19587
tags: [architecture, foundation, central]
---

# NeoBERT (central paper)

## Core idea
A modern, compact (250M) BERT-style **encoder** that ports a decade of LLM advances back into
the encoder world: RoPE positions, Pre-RMSNorm, SwiGLU FFN, an **optimal depth-to-width ratio**,
4,096-token context, RefinedWeb data, and a long, well-tuned MLM pre-training. It is a
plug-and-play replacement for BERT/RoBERTa and is SOTA on MTEB at its size.

## Key math / architecture details (250M-opt, from repo configs)
- **Shape:** 28 layers × hidden 768 × 12 heads (dim_head 64), intermediate 3072.
  The **deep-and-narrow** choice (28 layers rather than a wider/shallower 250M) is the
  "optimal depth-to-width ratio" claim — depth buys representational quality per parameter.
- **Positions:** **RoPE** applied to Q/K inside each block (no additive position embedding) →
  [[rope]]. Enables the 4,096 context extension.
- **Norm:** **Pre-RMSNorm** (`rms_norm: true`, `norm_eps: 1e-5`), final RMSNorm after the stack
  → [[rmsnorm]], [[pre-ln]].
- **FFN:** **SwiGLU** (`hidden_act: swiglu`), inner width 2/3-adjusted, rounded to a multiple of 8
  → [[swiglu]].
- **Attention:** fused no-bias `W_qkv`; xFormers memory-efficient attention by default, torch SDPA
  fallback, optional FlashAttention with sequence-packing → [[flash-attention]].
- **Tokenizer:** `google-bert` WordPiece, vocab ~30k. **Tokenizer reuse is exactly what SALT3 has
  to replace** for Vietnamese.
- **Objective:** MLM only (no NSP), **20% masking** (vs BERT 15%) → [[bert]].
- **Optimizer:** **AdamW**, lr 1e-4, **betas (0.9, 0.95)**, eps 1e-8, weight_decay 0.01 → [[adamw]].
  (The (0.9,0.95) pair is the LLaMA-2 recipe → [[llama]].)
- **Schedule:** warmup 2000 steps → **cosine** decay over ~900k steps → [[cosine-sgdr]].
- **Precision:** bf16 mixed precision, tf32 matmul, gradient clipping 1.0.
- **Data/compute:** RefinedWeb (2.8 TB), ~2.1T training tokens; context extended in a second phase.
- **Variant:** `NormNeoBERT` uses normalized "nGPT-style" updates and drops the final norm → [[ngpt]].

## Results / why it matters
Outperforms BERT-large, RoBERTa-large, NomicBERT, and ModernBERT on **MTEB** under an identical
fine-tuning protocol, despite 250M params; strong **GLUE** too. Establishes that an encoder
rebuilt with modern LLM components is markedly more parameter-efficient.

## How NeoBERT / SALT3 uses this
**NeoBERT is the base model SALT3 adapts.** SALT3 keeps the entire NeoBERT body (all 28 RoPE+
RMSNorm+SwiGLU blocks) frozen-in-architecture and *re-initializes only the token embedding matrix
and the LM-head (decoder)* for a pruned Vietnamese (ViDeBERTa/PhoBERT-derived) vocabulary, then
runs WSD-scheduled MLM continued pre-training on CulturaX-vi. Every hyperparameter the thesis
inherits — AdamW (0.9,0.95), cosine/cooldown shapes, 20% masking, bf16 — traces to this paper and
its configs (`conf/optimizer/adamw.yaml`, `conf/model/250M-opt.yaml`, `conf/scheduler/`).

## Relation: [[transformer]] [[bert]] [[rope]] [[swiglu]] [[rmsnorm]] [[adamw]] [[flash-attention]] [[ngpt]] [[modernbert]] [[nomicbert]]
