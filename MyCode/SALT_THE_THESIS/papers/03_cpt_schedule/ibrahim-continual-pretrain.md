---
title: Simple and Scalable Strategies to Continually Pre-train Large Language Models
authors: Ibrahim, Thérien, Gupta, Richter, Anthony, Lesort, Belilovsky, Rish
year: 2024
venue: TMLR 2024
arxiv: 2403.08763
url: https://arxiv.org/abs/2403.08763
tags: [cpt, scheduler, rewarming, forgetting]
---

# Continual pretraining: re-warm, re-decay, replay (Ibrahim et al.)

## Core idea
Three simple strategies let you **continue** pretraining on new data and **match full retraining from scratch**, without catastrophic forgetting: (1) **re-warm and re-decay** the learning rate, (2) **replay** a small fraction of previous data, (3) combine them. The first is the direct fix for the exact situation where a checkpoint's LR has **decayed to ~0**.

## Key details
- **LR re-warming + re-decaying:** on resuming, the LR must be **raised back up** (not continued from its decayed-to-near-zero value), then decayed again over the new phase. Continuing from LR≈0 means almost no weight updates; an *abrupt* LR jump without warmup hurts — hence a short warmup, then re-decay.
- **Replay:** mixing in a small % of the original distribution prevents collapse on previously-learned data under distribution shift.
- **Validated** on weak (En→En) and strong (En→De) shifts at 405M and 10B scale; combined recipe ≈ full retraining at a fraction of the cost.

## Results / why it matters
The canonical, practical recipe for continual pretraining; the standard citation for *why you must re-warm the LR* when extending training.

## How NeoBERT / SALT3 uses this
**Directly applicable.** SALT3's CPT checkpoints are **cooled to LR 0** (5B-token run, nb19). To train further (e.g. a fresh-data decay or a 4k extension) the LR **must be re-warmed** — exactly this paper's prescription, and what the WSD machinery does via a global-step-keyed warmup before re-decay ([[wsd-minicpm]], [[warmup]]). It is the reference for the thesis's **re-warming** subsection in the adaptation chapter, complementing the WSD/1-sqrt cooldown citations. The **replay** idea connects to the English-replay finding in [[emergent-cpt-language-adaptation]] as a forgetting guard / future work.

## Relation: [[wsd-minicpm]] [[hagele-cooldown]] [[warmup]] [[emergent-cpt-language-adaptation]] [[language-adaptation]]
