---
title: "Marian: Fast Neural Machine Translation in C++"
authors: Junczys-Dowmunt, Grundkiewicz, Dwojak, Hoang, Heafield, Neckermann, Seide, Germann, Fikri Aji, Bogoychev, Martins, Birch
year: 2018
venue: ACL 2018 (System Demonstrations)
arxiv: 1804.00344
url: https://arxiv.org/abs/1804.00344
tags: [nmt, translation, back-translation, tooling, anchor-mining]
---

# Marian / MarianMT — the NMT engine behind the mined translation anchors

## Core idea
Marian is a **self-contained neural machine translation toolkit written in C++**: high-performance
training and inference for Transformer/RNN NMT with no heavy framework dependency. It is the engine
that powers the **Helsinki-NLP OPUS-MT** model zoo — including `opus-mt-vi-en` (Vietnamese→English),
which SALT3 uses. In the HuggingFace ecosystem these models are exposed as **`MarianMTModel`**.

## Key details
- Optimized C++ with custom auto-diff, efficient batched beam search, INT/FP16 inference — very fast
  and memory-light at inference time.
- OPUS-MT: thousands of bilingual/multilingual translation models trained with Marian on the
  **OPUS** parallel corpora, released openly (`Helsinki-NLP/opus-mt-*`).
- Used widely for **back-translation** (translate to a pivot language and, optionally, back) as a
  data-augmentation / alignment-verification device.

## Results / why it matters
Marian made translation cheap enough to run at scale as a **preprocessing / mining tool** rather than
an end product. Reliable, fast, freely available direction-specific models (like vi→en) are exactly
what a large anchor-mining pass needs.

## How NeoBERT / SALT3 uses this
SALT3's **3-tier anchor mining** (the improvement over surface-overlap-only SALT) uses MarianMT
(`Helsinki-NLP/opus-mt-vi-en`) to **translate Vietnamese donor tokens to English and verify the
pair**, gated by a strict Vietnamese-orthography filter and a fastText check ([[fasttext-157lang]]).
This produces high-precision **vi→en translation-pair anchors** on top of same-surface-form and
shared-number anchors, enriching anchor coverage that the per-token SALT maps depend on ([[salt]]).
Cited in the thesis where the translation-verified anchors are introduced.

## Relation: [[salt]] [[focus]] [[fasttext-157lang]] [[videberta]]
