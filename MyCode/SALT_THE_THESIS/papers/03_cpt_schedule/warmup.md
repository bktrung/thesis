---
title: "On the Variance of the Adaptive Learning Rate and Beyond (RAdam)"
authors: Liu, Jiang, He, Chen, Liu, Gao, Han
year: 2020
venue: ICLR 2020
arxiv: 1908.03265
url: https://arxiv.org/abs/1908.03265
tags: [cpt, scheduler, warmup, optimizer]
---

# Warmup (why linear warmup exists)

## Core idea
Early in Adam training, the **adaptive second moment `v_t` is estimated from very few samples**, so
its variance is huge and the effective per-parameter LR is wildly unreliable — taking full-size steps
then can destabilize training. A **learning-rate warmup** (start near 0, ramp up over the first few
hundred/thousand steps) is a *variance-reduction* fix: it holds steps small until `v_t` is trustworthy.
RAdam derives this and makes the warmup *automatic*; the practical takeaway for everyone else is
**"linear warmup is principled, not a hack."**

## Key math / architecture details
- Linear warmup: `lr(t) = peak · t / W` for `t < W`, then hand off to the main schedule.
- RAdam computes a **rectification term** `r_t` from the estimated degrees of freedom of `v_t`; when
  the variance is untrustworthy it falls back to (un-adapted) SGD-with-momentum, equivalent to an
  automatically-tuned warmup.
- Warmup also interacts with **post-norm vs pre-norm**: pre-norm Transformers need *less* warmup
  ([[pre-ln]]), but some warmup still stabilizes the adaptive-LR variance at the start of a new phase.
- For **continued pre-training**, a *short* warmup re-stabilizes the optimizer when entering a new
  data/LR regime — but must not re-warm on every resume (the WSD global-step keying fix → [[wsd-minicpm]]).

## Results / why it matters
Explains and removes the need to hand-tune warmup length for Adam; validates linear warmup as the
default opening of essentially every Transformer LR schedule.

## How NeoBERT / SALT3 uses this
NeoBERT pretraining warms up over **2000 steps** (`conf/scheduler`). SALT3's CPT keeps a **short
warmup = 2% of the planned ceiling, clamped to [20, 500]** (`compute_warmup_steps` in
`salt3_staged_schedule.py`) so even a tiny dry-run or a 10M-doc ceiling both finish warmup within the
horizon. Critically, SALT3 keys warmup on the **global step**, so `wsd_lr_at` ramps once and then
stays flat — a resumed session does **not** re-warm (avoiding the LR spike that would perturb the
carried AdamW moments → [[adamw]]). This is the warmup theory applied to *staged* continued pretraining.

## Relation: [[wsd-minicpm]] [[cosine-sgdr]] [[pre-ln]] [[adamw]]
