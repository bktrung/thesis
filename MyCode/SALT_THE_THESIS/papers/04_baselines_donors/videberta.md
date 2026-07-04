---
title: "ViDeBERTa: A powerful pre-trained language model for Vietnamese"
authors: Tran, Pham, Nguyen, Hy, Vu
year: 2023
venue: EACL 2023 Findings
arxiv: 2301.10439
url: https://arxiv.org/abs/2301.10439
tags: [donor, vietnamese, primary-donor]
---

# ViDeBERTa (primary Vietnamese donor)

## Core idea
A monolingual **Vietnamese DeBERTa**. Built on the DeBERTa architecture and trained on a large
Vietnamese corpus, it reaches strong results with far fewer parameters than PhoBERT (ViDeBERTa-base
86M ≈ 23% of PhoBERT-large, matching or beating it). Crucially for SALT3, it uses a **SentencePiece
Unigram** tokenizer — *not* word-segmented — which aligns better with NeoBERT's subword style.

## Key math / architecture details
- **Architecture:** DeBERTa-family (disentangled attention; xsmall / **base 86M** / large).
- **Tokenizer:** **SentencePiece with the Unigram algorithm** — vocab entries are unique pieces with
  log-probability scores. This is why the thesis can **prune** the donor vocab by matching piece
  strings exactly (Unigram pieces are unique) — see `rebuild_vocab_maps()` in
  `salt3_decoder_variants.py` and the `videberta_load_probe` forensics.
- **No word segmentation** (unlike PhoBERT → [[phobert]]) → token boundaries are closer to NeoBERT's
  WordPiece, simplifying anchor alignment.
- Evaluated on POS, NER, QA for Vietnamese.

## Results / why it matters
Parameter-efficient Vietnamese encoder; its **Unigram tokenizer + raw-text (unsegmented)** design is
exactly what makes it the convenient donor for transferring into NeoBERT's subword space.

## How NeoBERT / SALT3 uses this
ViDeBERTa is SALT3's **primary embedding donor**. The init mines anchor pairs (vi token ↔ neo token),
prunes the ViDeBERTa vocabulary to the target Vietnamese set, and builds NeoBERT-space embeddings as a
**sparsemax-weighted average of ViDeBERTa rows** (or via Procrustes) — `videberta_token`/`neobert_token`
columns in the anchor CSVs, `new_to_old` mapping to original ViDeBERTa rows, and the whole
`salt3_videberta_load_probe.py` / `18_videberta_load_probe` chain. Much of the thesis's debugging
(`06_videberta_load_forensics`) is about loading this donor correctly. → [[salt]], [[sparsemax]],
[[procrustes]].

## Relation: [[phobert]] [[salt]] [[sparsemax]] [[procrustes]] [[neobert]] [[culturax]]
