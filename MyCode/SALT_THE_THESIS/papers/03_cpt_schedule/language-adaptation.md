---
title: "As Good as New. How to Successfully Recycle English GPT-2 to Make Models for Other Languages"
authors: de Vries, Nissim
year: 2021
venue: ACL-IJCNLP 2021 Findings
arxiv: 2010.02559
url: https://arxiv.org/abs/2010.02559
tags: [cpt, crosslingual, adaptation]
---

# Cross-lingual model recycling (language adaptation)

## Core idea
You don't need to pretrain a new-language model from scratch — **recycle** an existing English model.
Keep the **Transformer body**, learn a **new-language embedding layer** (new tokenizer/vocabulary),
and continue training. The body's language-agnostic structure transfers; only the lexical layer needs
to be (re)learned. The strongest variant **retrains embeddings while adapting the transformer**,
recovering near-from-scratch quality at a fraction of the cost.

## Key math / architecture details
- **Stage 1 — new embeddings:** freeze the transformer body, train *only* the new embedding matrix on
  target-language text so the lexical layer aligns to the existing representation space.
- **Stage 2 — joint adaptation:** unfreeze and continue training body + embeddings on target text.
- A good **embedding init** shortens stage 1 dramatically — this is precisely where WECHSEL/FOCUS/
  SALT plug in ([[wechsel]], [[focus]], [[salt]]).
- Confirms catastrophic-forgetting of English is acceptable when the goal is a monolingual target
  model, and that the body's syntactic/structural knowledge is reusable across languages.

## Results / why it matters
Recycled models reach quality comparable to from-scratch training with far less compute, across
several languages — the empirical license for the entire "adapt an English model to language X"
program that SALT3 instantiates for Vietnamese.

## How NeoBERT / SALT3 uses this
This is the **blueprint SALT3 follows**: recycle the English **NeoBERT** body, install a
**Vietnamese embedding/decoder** built by the SALT init, and **continue MLM pretraining** on
CulturaX-vi. The thesis even mirrors the **freeze-then-unfreeze** idea — see
`14_freeze_align_encoder_decoder_init` (freeze/align encoder–decoder init) — and the
embedding-only vs full-body adaptation tradeoffs. SALT3's contribution over this paper is the
**quality of the init** (anchor-sparsemax + freq-bias, [[salt]], [[freq-bias-init]]) and a
**resumable WSD schedule** ([[wsd-minicpm]]) for the staged adaptation. General CPT rationale →
[[dont-stop-pretraining]].

## Relation: [[dont-stop-pretraining]] [[wechsel]] [[focus]] [[salt]] [[wsd-minicpm]] [[neobert]]
