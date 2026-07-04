---
title: "Learning Word Vectors for 157 Languages"
authors: Grave, Bojanowski, Gupta, Joulin, Mikolov
year: 2018
venue: LREC 2018
arxiv: 1802.06893
url: https://arxiv.org/abs/1802.06893
tags: [embeddings, fasttext, static-vectors, multilingual, vietnamese]
---

# fastText 157-language vectors (the `cc.vi.300` Vietnamese vectors)

## Core idea
Releases **pre-trained fastText word vectors for 157 languages**, trained on **Common Crawl +
Wikipedia** with the subword model of [[fasttext-subword]]. This is the *distribution* paper: it is
where the widely-used `cc.<lang>.300.bin` / `.vec` files come from — including the **Vietnamese**
model `cc.vi.300` that SALT3 loads.

## Key details
- CBOW with position weights, 300 dimensions, character n-grams of length 5, window size 5, 10
  negatives — a fixed recipe applied uniformly across all 157 languages.
- Trained on the union of **Common Crawl** and **Wikipedia** per language → far larger and broader
  coverage than Wikipedia-only vectors.
- Ships as `.bin` (retains subword n-grams → can vectorize OOV words) and `.vec` (words only).
- Evaluated on word analogy datasets built/translated for many languages.

## Results / why it matters
Made high-quality static embeddings available off-the-shelf for low- and mid-resource languages
(Vietnamese included), removing the need to train your own static space. The `.bin` format's ability
to compose OOV vectors from subwords is essential when the tokens you must embed are arbitrary
subword pieces rather than dictionary words.

## How NeoBERT / SALT3 uses this
`cc.vi.300.bin` is the concrete **static similarity space** in the SALT3 pipeline (per
[[salt3-method-verified]]): every Vietnamese donor token gets a fastText vector, and cosine
similarities in this space drive the 3-tier anchor mining and the **sparsemax** anchor weights used
to build the per-token embedding/decoder maps ([[salt]], [[sparsemax]], [[focus]]). Using the `.bin`
model guarantees a vector even for rare or subword-fragmented tokens.

## Relation: [[fasttext-subword]] [[salt]] [[sparsemax]] [[focus]] [[videberta]]
