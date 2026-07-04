---
title: "LLaMA: Open and Efficient Foundation Language Models / Llama 2: Open Foundation and Fine-Tuned Chat Models"
authors: Touvron et al. (Meta AI)
year: 2023
venue: preprints (Meta AI)
arxiv: "2302.13971 (LLaMA); 2307.09288 (Llama 2)"
url: https://arxiv.org/abs/2307.09288
tags: [baseline, recipe, architecture]
---

# LLaMA / LLaMA-2 (the modern recipe NeoBERT borrows)

## Core idea
LLaMA showed that a **clean, well-tuned decoder** with a few specific component choices, trained on a
lot of public data, is extremely strong and efficient. Its component set — **RoPE, SwiGLU,
RMSNorm/pre-norm, AdamW** — became the de-facto "modern Transformer" recipe that encoders like NeoBERT
then imported. LLaMA-2 refined the training recipe (incl. the AdamW betas).

## Key math / architecture details
- **Pre-RMSNorm** ([[rmsnorm]], [[pre-ln]]), **RoPE** positions ([[rope]]), **SwiGLU** FFN with 2/3
  width ([[swiglu]]) — exactly NeoBERT's encoder choices.
- **AdamW** with **betas (0.9, 0.95)**, weight_decay 0.01, gradient clipping 1.0, cosine LR decay →
  [[adamw]], [[cosine-sgdr]]. The lower `β₂=0.95` tracks the second moment more responsively.
- BPE/SentencePiece tokenizer; trained on trillions of tokens; depth/width scaled by size.
- LLaMA-2: 2T-token pretraining, 4k context, grouped-query attention at larger sizes.

## Results / why it matters
Defined the open modern-LLM recipe; its hyperparameters propagated across the field. When a paper says
"standard modern Transformer settings," it usually means *these*.

## How NeoBERT / SALT3 uses this
NeoBERT explicitly adopts the **LLaMA recipe** for an encoder: RoPE + SwiGLU + Pre-RMSNorm + AdamW.
The thesis code names it directly — `salt3_staged_schedule.py` comments that **betas (0.9, 0.95)** is
"the NeoBERT/LLaMA-2 AdamW recipe" and keeps it for CPT so the optimizer's preconditioner matches the
regime the body was pretrained under. So LLaMA/LLaMA-2 is the upstream source of SALT3's optimizer and
architecture constants. → [[neobert]], [[adamw]].

## Relation: [[neobert]] [[rope]] [[swiglu]] [[rmsnorm]] [[adamw]] [[cosine-sgdr]]
