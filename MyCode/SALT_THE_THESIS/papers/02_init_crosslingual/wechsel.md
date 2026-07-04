---
title: "WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models"
authors: Minixhofer, Paischer, Rekabsaz
year: 2022
venue: NAACL 2022
arxiv: 2112.06598
url: https://arxiv.org/abs/2112.06598
tags: [init, crosslingual, lineage]
---

# WECHSEL (lineage root)

## Core idea
To move a monolingual model (e.g. English RoBERTa/GPT-2) to a new language, **swap the tokenizer**
and **initialize the new subword embeddings smartly** instead of from scratch. WECHSEL bridges the
two subword spaces through a shared **bilingual static word-embedding** space (fastText), so a
target token's embedding is a similarity-weighted blend of *source* token embeddings that mean
similar things.

## Key math / architecture details
1. **Bilingual static space:** align source and target fastText embeddings into one space using a
   bilingual dictionary (orthogonal/Procrustes-style mapping → [[procrustes]]).
2. **Subword → static vector:** represent each subword token (source and target) as a (length-
   weighted) composition of the static word vectors it appears in. Gives every token a vector in the
   shared space.
3. **Transfer by similarity:** for each **target** subword `t`, compute cosine similarity to all
   **source** subwords in the shared space; the new embedding is the **softmax-similarity-weighted
   average** of source model embeddings:
   `E_target[t] = Σ_s softmax(sim(t,s)/τ)_s · E_source[s]`.
4. Special/overlapping tokens handled directly; the rest get this convex combination.

## Results / why it matters
Beats from-scratch training of comparable models with **up to 64× less compute**, across French,
German, Chinese, Swahili — strongest for low-resource languages. Established the template:
*static-embedding similarity → weighted average of donor rows.*

## How NeoBERT / SALT3 uses this
WECHSEL is the **root of the SALT3 init lineage**. SALT3 keeps WECHSEL's core move — a target
embedding = similarity-weighted average of donor embeddings — but changes (a) the *donor* from a
generic source model to a **target-language PLM** (ViDeBERTa/PhoBERT, like SALT → [[salt]]), (b) the
weighting from softmax to **sparsemax over mined anchors** (like FOCUS → [[focus]], [[sparsemax]]),
and (c) adds a **frequency-bias decoder** ([[freq-bias-init]]). FOCUS → SALT → SALT3 are all
refinements of this paper's idea.

## Relation: [[focus]] [[salt]] [[sparsemax]] [[procrustes]] [[freq-bias-init]]
