---
title: "Should You Mask 15% in Masked Language Modeling?"
authors: Wettig, Gao, Zhong, Chen
year: 2023
venue: EACL 2023
arxiv: 2202.08005
url: https://arxiv.org/abs/2202.08005
tags: [mlm, pretraining-objective, masking, bert]
---

# Mask ratio — 15% is not sacred (why NeoBERT masks more)

## Core idea
Revisits BERT's default **15% masking rate** for masked language modeling ([[bert]]) and shows it is
**not optimal**. Larger models can be trained effectively with **substantially higher masking rates**
(e.g. **~20–40%**), which increases the number of prediction targets per sequence and can improve
downstream performance and training efficiency.

## Key details
- Disentangles two effects of masking: the **corruption rate** (how much the input is degraded) vs.
  the **prediction rate** (how many tokens the model must predict). Higher masking raises the useful
  prediction signal per example.
- Also revisits BERT's **80/10/10** mask/replace/keep split and argues the elaborate corruption is
  largely unnecessary — simple masking works.
- Optimal rate scales with **model size and objective**: bigger models tolerate/benefit from higher
  masking.

## Results / why it matters
Directly overturns the "always 15%" folklore. Modern efficient encoders adopt higher masking rates as
a cheap way to get more learning signal per token. This is part of why **ModernBERT** ([[modernbert]])
and **NeoBERT** ([[neobert]]) depart from 15%.

## How NeoBERT / SALT3 uses this
NeoBERT trains its MLM objective with a **higher masking rate (20%)** rather than BERT's 15%, and this
paper is the justification cited in the thesis's NeoBERT/CPT background. It carries directly into
SALT3: the **freeze-align** stage and the **staged WSD continued pre-training** on CulturaX both use
**MLM at 20% masking** ([[salt3-method-verified]], [[wsd-minicpm]], [[culturax]]) — so this note
grounds the objective's hyperparameter choice.

## Relation: [[bert]] [[neobert]] [[modernbert]] [[roberta]]
