---
title: "Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-lingual Transfer"
authors: Lee, Hong, Moon, Lim
year: 2025
venue: ACL 2025 Findings
arxiv: 2505.10945
url: https://arxiv.org/abs/2505.10945
tags: [init, crosslingual, lineage, origin-method]
---

# SALT — Semantic Aware Linear Transfer (the method the thesis improves)

## Core idea
**SALT = Semantic Aware Linear Transfer.** Instead of replacing a source LLM's vocabulary with a
randomly/heuristically initialized target vocabulary, **recycle the embeddings of an existing
target-language pre-trained model (PLM)** and *linearly* transfer their representational strength
into the source LLM's embedding space. The key device: per-token **regression lines** fit on the
**overlap** of source and target vocabularies, then used to place the **non-overlapping** target
tokens.

## Key math / architecture details
- **Overlap as supervision:** for tokens shared by the source LLM and the target PLM, you have
  *paired* embeddings `(e_src, e_tgt)`. These pairs define the transfer.
- **Unique regression lines:** SALT derives **per-token / similarity-local linear maps** ("unique
  regression lines based on the similarity in the overlap") rather than one global matrix — a
  *locally linear* mapping from the target-PLM space into the source-LLM space.
- **Non-overlap transfer:** embeddings for non-overlapping target tokens are produced by applying the
  appropriate (similarity-weighted) regression line to the target-PLM embedding, importing the PLM's
  semantics into the LLM's space.
- **"Recycling":** the target PLM is treated as a free source of language-specific signal — no
  retraining of the donor.

## Results / why it matters
Lower adaptation loss and **faster convergence** during language adaptation, better cross-lingual
understanding than competing init methods, and scalability across architectures. It is the
state-of-the-art "recycle a target PLM via linear transfer" formulation — and the **named method the
thesis extends to "SALT3."**

## How NeoBERT / SALT3 uses this  — *What we changed vs origin SALT*
SALT3 takes SALT's "recycle a target PLM by locally-linear transfer over anchors" and adapts it from
a decoder-LLM setting to the **NeoBERT encoder + Vietnamese** setting, with concrete improvements:
1. **Donor = Vietnamese PLM:** ViDeBERTa / PhoBERT ([[videberta]], [[phobert]]) supply the recycled
   embeddings, mapped into NeoBERT space.
2. **Sparsemax anchor weighting (from FOCUS):** the local combination uses **sparsemax over mined
   anchors** (`sparsemax(ft @ anchor_ftᵀ)`), a 3-tier anchor set (verified translations, shared
   numbers, mined pairs) rather than only vocab-string overlap → [[focus]], [[sparsemax]].
3. **Global emb→dec map for the LM head:** the decoder is built by fitting a **global NeoBERT
   embedding→decoder linear map** `M = lstsq(E_neo, W_neo)` and applying it to SALT embeddings
   (`W = E_salt @ M`) — see `scripts/test_decoder_global_map_and_freq_bias.py`,
   `salt3_decoder_variants.py`.
4. **Frequency-bias decoder:** the LM-head **bias** is set to the **Vietnamese unigram
   log-frequency** (the prior original SALT/FOCUS left at zero) → [[freq-bias-init]].
5. **Procrustes alternative arm** for the embedding map → [[procrustes]].
6. Careful **norm/scale matching** (`meannorm`) and special-token plumbing so comparisons isolate the
   transfer method, then **WSD continued pre-training** → [[wsd-minicpm]].

## Relation: [[wechsel]] [[focus]] [[sparsemax]] [[procrustes]] [[freq-bias-init]] [[videberta]] [[phobert]] [[neobert]]
