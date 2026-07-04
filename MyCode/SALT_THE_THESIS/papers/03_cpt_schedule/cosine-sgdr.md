---
title: "SGDR: Stochastic Gradient Descent with Warm Restarts"
authors: Loshchilov, Hutter
year: 2017
venue: ICLR 2017
arxiv: 1608.03983
url: https://arxiv.org/abs/1608.03983
tags: [cpt, scheduler, cosine]
---

# Cosine decay / SGDR

## Core idea
Anneal the learning rate along a **cosine curve** from peak to ~0 over training, optionally with
periodic **warm restarts** (jump back to peak and decay again). The cosine shape spends a lot of
steps at moderate-to-low LR, which empirically converges better than step decay and is smooth/
parameter-light. The non-restart cosine decay is now the default LR schedule for Transformer
pretraining.

## Key math / architecture details
- Single cosine decay over `T` steps:
  `lr(t) = lr_min + ½(lr_max − lr_min)(1 + cos(π t/T))`.
- **Warm restarts:** reset `t` every `T_i` steps (with `T_i` often growing geometrically); each
  restart can escape sharp minima and ensemble-like averages improve generalization.
- A `final_ratio` (η_min = lr·ratio) keeps a small floor LR at the end rather than exactly 0.
- Contrast with WSD: cosine couples the schedule to a **fixed total length** `T`; resuming past `T`
  is ill-defined → motivates the WSD plateau for continual training ([[wsd-minicpm]]).

## Results / why it matters
Faster convergence and better final accuracy than step schedules; the cosine *shape* survives as the
default decay/cooldown curve even inside WSD.

## How NeoBERT / SALT3 uses this
**Two places.** (1) NeoBERT's *original* pretraining uses **cosine decay** (`conf/scheduler/*.yaml`:
`decay: cosine`, warmup 2000 → decay over ~900k; the HF scheduler in
`NeoBERT/src/neobert/scheduler/scheduler.py` builds `LinearLR(warmup) → CosineAnnealingLR(η_min =
lr·final_ratio)`). (2) SALT3's CPT reuses the cosine **as one of the cooldown shapes**
(`cooldown_lr_at` = `peak·½(1+cos(π·frac))`), the alternative to 1-sqrt ([[hagele-cooldown]]). So
cosine is both the inherited baseline schedule and a selectable cooldown in the thesis's WSD machinery.

## Relation: [[wsd-minicpm]] [[hagele-cooldown]] [[warmup]] [[adamw]] [[neobert]]
