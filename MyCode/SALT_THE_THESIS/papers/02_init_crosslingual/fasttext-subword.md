---
title: "Enriching Word Vectors with Subword Information"
authors: Bojanowski, Grave, Joulin, Mikolov
year: 2017
venue: TACL 2017 (Vol. 5)
arxiv: 1607.04606
url: https://arxiv.org/abs/1607.04606
tags: [embeddings, fasttext, static-vectors, subword]
---

# fastText — subword word vectors (the static space anchors are scored in)

## Core idea
Extends the skip-gram model (word2vec) by representing each word as a **bag of character
n-grams** plus the whole-word token. A word's vector is the **sum of its subword n-gram
vectors**, so morphology is shared across words and **out-of-vocabulary (OOV) words still get a
vector** by composing their n-grams. This is the "fastText" static embedding.

## Key math / details
- Each word `w` → set of character n-grams `G_w` (e.g. 3–6 grams, with `<` `>` word boundaries).
- Word representation: `z_w = Σ_{g ∈ G_w} v_g` (sum of n-gram vectors `v_g`).
- Skip-gram scoring with negative sampling: `s(w, c) = Σ_{g ∈ G_w} v_g · u_c`.
- Advantages: robust for **morphologically rich languages**, rare words, and unseen words; fast to
  train (C++, hashing of n-grams).
- Produces **300-dim static vectors** in the released models.

## Results / why it matters
Beats word2vec/GloVe on word-similarity and analogy tasks, especially for rare/inflected words and
for languages with rich morphology. Became the de-facto **static (non-contextual) embedding** for
computing token-level semantic similarity when a lightweight, OOV-robust space is needed.

## How NeoBERT / SALT3 uses this
The SALT3 anchor-mining and per-token transfer both operate in a **static fastText embedding
space**: candidate Vietnamese tokens are scored by **cosine similarity in fastText space** against
anchor tokens to build the sparsemax weights ([[sparsemax]]) and the local least-squares maps
([[salt]]). Subword composition is what lets fastText assign a usable vector to essentially any
Vietnamese wordpiece, including ones absent from the anchor set. The specific vectors used are the
pre-trained Vietnamese vectors from [[fasttext-157lang]] (`cc.vi.300`).

## Relation: [[fasttext-157lang]] [[salt]] [[focus]] [[sparsemax]] [[wechsel]]
