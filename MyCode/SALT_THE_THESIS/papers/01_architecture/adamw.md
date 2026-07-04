---
title: Decoupled Weight Decay Regularization
authors: Loshchilov, Hutter
year: 2019
venue: ICLR 2019
arxiv: 1711.05101
url: https://arxiv.org/abs/1711.05101
tags: [architecture, optimizer]
---

# AdamW

## Core idea
In adaptive optimizers, **L2 regularization and weight decay are not equivalent** (they are for
plain SGD). Folding the penalty into the gradient lets Adam's per-parameter adaptive scaling shrink
the effective decay on large-gradient weights. **AdamW** fixes this by *decoupling* weight decay:
apply it directly to the weights, outside the adaptive moment update.

## Key math / architecture details
- Adam update: `m_t, v_t` are EMA of gradient and squared gradient with decays `β₁, β₂`; step is
  `θ ← θ − lr · m̂_t / (√v̂_t + ε)`.
- **AdamW** adds a separate decay term applied to the parameter itself:
  `θ ← θ − lr · ( m̂_t/(√v̂_t+ε) + λ·θ )`,
  so the decay `λ` is independent of the gradient magnitude and of `v_t`.
- Decouples the optimal `lr` and `λ` (they no longer trade off), making tuning easier and improving
  generalization.

## Results / why it matters
Better generalization and more stable training than Adam-with-L2; the universal default for
training Transformers/LLMs.

## How NeoBERT / SALT3 uses this
NeoBERT's optimizer is **AdamW**: `conf/optimizer/adamw.yaml` → lr 1e-4, **betas (0.9, 0.95)**,
eps 1e-8, weight_decay 0.01. The lower `β₂=0.95` (vs Adam's 0.999) tracks the second moment more
responsively — the LLaMA-2 recipe ([[llama]]). SALT3's continued pre-training **deliberately keeps
AdamW external and carries its moments across staged sessions** so the preconditioner the body was
trained under is preserved (`build_wsd_optimizer` in `salt3_staged_schedule.py`). This is why the
WSD LR is driven by a callback rather than re-creating the optimizer — re-init would throw away
`m_t, v_t` → [[wsd-minicpm]].

## Relation: [[neobert]] [[llama]] [[wsd-minicpm]] [[cosine-sgdr]]
