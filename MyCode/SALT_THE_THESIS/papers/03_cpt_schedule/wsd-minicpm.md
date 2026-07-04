---
title: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies"
authors: Hu, Tu, Han, He, Cui, Long, et al. (OpenBMB / Tsinghua)
year: 2024
venue: preprint (COLM 2024)
arxiv: 2404.06395
url: https://arxiv.org/abs/2404.06395
tags: [cpt, scheduler, wsd]
---

# WSD scheduler (MiniCPM)

## Core idea
The **Warmup-Stable-Decay (WSD)** learning-rate schedule: (1) **warmup** 0→peak, (2) a long
**stable** phase holding a *constant* high LR, (3) a short, rapid **decay** ("cooldown") to near
zero only at the very end. Decoupling the stable plateau from the decay means you can **continue
training from any point on the plateau** and only pay the decay cost when you want a deliverable
checkpoint — ideal for continual / staged training and for studying scaling without retraining.

## Key math / architecture details
- **Warmup:** linear `0 → peak` over `W` steps.
- **Stable:** LR `= peak`, constant, for as long as you keep training (the plateau).
- **Decay/cooldown:** rapidly anneal `peak → ~0` over a short window (cosine, linear, or 1-sqrt →
  [[hagele-cooldown]]). The sharp loss drop happens during this phase.
- **Why it beats cosine for CPT:** cosine bakes the total step count into the schedule, so resuming
  past the planned end **re-warms or mis-scales**; WSD's plateau is **step-count-agnostic**, so a
  resumed session just continues at peak with **no re-warm spike**.
- MiniCPM also reports a higher **compute-optimal data/model ratio** than Chinchilla.

## Results / why it matters
WSD matches or beats cosine while enabling continual pretraining and cheap scaling-law studies; now
a standard recipe for models meant to be trained in stages.

## How NeoBERT / SALT3 uses this
WSD is the **schedule SALT3's continued pre-training is built on**. `salt3_staged_schedule.py`
implements it as a **TrainerCallback keyed on the GLOBAL optimizer step**: `wsd_lr_at(step)` does
linear warmup then **flat peak forever**, and a separate **cooldown** branch (`cooldown_lr_at` cosine
or `onesqrt_lr_at` 1-sqrt) anneals to 0 only at a milestone. This is *exactly* the WSD decoupling —
chosen so a resumed/continued Colab session **never re-warms** (the file's docstring says so), and so
the external **AdamW moments carry across sessions** ([[adamw]]). Warmup is 2% of the planned ceiling,
clamped [20,500] (`compute_warmup_steps`). Cooldown shape detail → [[hagele-cooldown]]; warmup
rationale → [[warmup]].

## Relation: [[hagele-cooldown]] [[cosine-sgdr]] [[warmup]] [[adamw]] [[dont-stop-pretraining]] [[neobert]]
