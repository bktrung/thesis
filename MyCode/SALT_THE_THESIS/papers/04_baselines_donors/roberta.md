---
title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
authors: Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer, Stoyanov
year: 2019
venue: preprint (FAIR)
arxiv: 1907.11692
url: https://arxiv.org/abs/1907.11692
tags: [baseline, encoder]
---

# RoBERTa

## Core idea
BERT was **undertrained**. RoBERTa keeps BERT's architecture but fixes the *recipe*: more data,
longer training, bigger batches, **no NSP**, **dynamic masking**, and a byte-level BPE tokenizer.
Pure recipe changes → large gains, establishing that pretraining data/compute/objective choices
dominate architecture tweaks at this scale.

## Key math / architecture details
- **Drop NSP:** train on full-sentence MLM only; matches or beats BERT.
- **Dynamic masking:** re-sample the masked positions each epoch (vs BERT's static mask).
- **Bigger everything:** 160GB text, 8k batch size, longer schedules, 50k byte-level BPE vocab.
- Same Transformer encoder (GELU, post-norm, learned positions) — *no architecture change*.

## Results / why it matters
SOTA on GLUE/SQuAD/RACE on release; became the standard strong encoder baseline and the architecture
template for many monolingual models — including **PhoBERT** ([[phobert]]).

## How NeoBERT / SALT3 uses this
RoBERTa is a **headline baseline NeoBERT outperforms** (NeoBERT beats RoBERTa-large on MTEB at 250M).
It is also doubly relevant to SALT3 because the **donor PhoBERT is a RoBERTa** trained for Vietnamese —
so RoBERTa's design (BPE vocab, GELU, learned positions) describes one of the two embedding donors the
SALT init recycles. The NSP-free, MLM-only objective RoBERTa validated is the one NeoBERT (and thus
SALT3's CPT) uses. → [[bert]], [[neobert]], [[phobert]].

## Relation: [[bert]] [[neobert]] [[phobert]] [[modernbert]]
