---
title: "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
authors: Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith
year: 2020
venue: ACL 2020
arxiv: 2004.10964
url: https://arxiv.org/abs/2004.10964
tags: [cpt, adaptation, foundation]
---

# Don't Stop Pretraining (DAPT/TAPT)

## Core idea
A general pretrained model is not the end — a **second phase of self-supervised pretraining** on
text closer to the target distribution gives consistent downstream gains. Two flavors:
**DAPT** (Domain-Adaptive Pre-Training: continue MLM on in-domain corpora) and **TAPT**
(Task-Adaptive: continue MLM on the task's own unlabeled text). They stack.

## Key math / architecture details
- Same MLM objective as pretraining, just on a new corpus, *before* fine-tuning.
- DAPT helps most when the target domain is **far** from the original pretraining distribution.
- TAPT is cheap (small task corpus) and complementary to DAPT.
- Establishes "continued pre-training" (CPT) as a distinct, effective stage — the conceptual basis
  for adapting across **languages**, not just domains.

## Results / why it matters
Gains across 4 domains × 8 tasks, in both high- and low-resource settings; the canonical reference
that *continuing* MLM pretraining is worthwhile. Cross-lingual adaptation (English→Vietnamese) is the
extreme "domain shift" case of this idea.

## How NeoBERT / SALT3 uses this
SALT3's whole second stage **is** continued pre-training: after the cross-lingual embedding init, it
runs **MLM CPT on CulturaX-vi** ([[culturax]]) to adapt the English NeoBERT body to Vietnamese. DAPT
is the justification that this stage produces real gains and that the model should not just be
fine-tuned cold. The thesis specializes DAPT in two ways the original doesn't cover: a **new
vocabulary/tokenizer** (handled by the init, [[salt]]) and a **WSD schedule** designed for *staged,
resumable* CPT ([[wsd-minicpm]]). Language adaptation specifics → [[language-adaptation]].

## Relation: [[wsd-minicpm]] [[language-adaptation]] [[salt]] [[culturax]] [[bert]]
