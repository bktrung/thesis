---
title: Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations
authors: Hägele, Bakouch, Kosson, Ben Allal, Von Werra, Jaggi
year: 2024
venue: NeurIPS 2024 (Spotlight)
arxiv: 2405.18392
url: https://arxiv.org/abs/2405.18392
tags: [cpt, scheduler, cooldown, scaling-laws]
---

# Constant-LR + cooldown / the 1-sqrt shape

## Core idea
A **constant learning rate followed by a short cooldown** matches cosine-schedule loss while being
far more flexible: you can train the *same* model for *different* durations and get a usable model at
each, instead of committing to one cosine length. This makes scaling-law studies cheap (reusable
runs) and is the rigorous backing for the WSD plateau. The paper also finds the **cooldown shape
matters**, with **1-sqrt** the strongest.

## Key math / architecture details
- During cooldown over `T_c` steps with fraction `f = t/T_c ∈ [0,1]`:
  - **cosine:** `lr(f) = peak · ½(1 + cos(π f))`
  - **1-sqrt:** `lr(f) = peak · (1 − √f)`  ← drops fast early, then flattens near 0
  - linear: `lr(f) = peak · (1 − f)`
- **Why 1-sqrt:** at a fixed cooldown budget it spends more steps at *low* LR (fine-grained
  convergence) while still leaving the plateau quickly — best downstream quality of the shapes tested.
- Constant+cooldown **scales as predictably as cosine**; cooldown length ~ a small fraction of total.
- **SWA** (stochastic weight averaging) along the trajectory gives a free extra boost.

## Results / why it matters
Spotlight result: you don't need a fixed-length cosine schedule to be compute-optimal; constant+1-sqrt
cooldown is the modern recipe for flexible / continual training and what makes WSD principled.

## How NeoBERT / SALT3 uses this
**Cited verbatim in the repo.** `salt3_staged_schedule.py::onesqrt_lr_at` implements
`peak·(1 − √(t/T_c))` with the comment: *"the strongest cooldown shape for downstream quality at a
fixed decay budget over fresh data (Hägele et al. 2024, 2405.18392)."* SALT3 offers both
`COOLDOWN_SHAPES = {"cosine", "1-sqrt"}` and sizes the cooldown at ~10% of budget-so-far
(`cooldown_steps_for`, cap 1000). This is the paper that justifies SALT3 running a **flat plateau +
short 1-sqrt cooldown over fresh CulturaX-vi data** instead of NeoBERT's original cosine, while
staying compute-comparable.

## Relation: [[wsd-minicpm]] [[cosine-sgdr]] [[warmup]] [[neobert]]
