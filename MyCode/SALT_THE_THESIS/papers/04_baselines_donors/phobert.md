---
title: "PhoBERT: Pre-trained language models for Vietnamese"
authors: Nguyen, Nguyen
year: 2020
venue: EMNLP 2020 Findings
arxiv: 2003.00744
url: https://arxiv.org/abs/2003.00744
tags: [donor, vietnamese, baseline]
---

# PhoBERT (Vietnamese donor + baseline)

## Core idea
The first large-scale **monolingual Vietnamese** pretrained LM, built on **RoBERTa**. Two key
Vietnamese-specific choices: **word segmentation** (VnCoreNLP) so the model operates on Vietnamese
*words* (which are often multi-syllable, space-separated), and a Vietnamese **BPE** vocabulary.
Beats multilingual XLM-R on Vietnamese tasks.

## Key math / architecture details
- **Architecture:** RoBERTa-base / -large ([[roberta]]) — GELU, learned positions, post-norm, MLM-only.
- **Preprocessing:** VnCoreNLP word segmentation *before* BPE — segments syllables into words, so
  token boundaries differ from a raw-text tokenizer. (A consideration when aligning to NeoBERT's
  WordPiece, which is *not* word-segmented.)
- **Vocabulary:** Vietnamese-specific byte-pair encoding, ~64k.
- Strong on POS, dependency parsing, NER, NLI for Vietnamese.

## Results / why it matters
Long the standard Vietnamese encoder and the **primary comparison point** for any new Vietnamese model;
the thesis evaluates "PhoBERT vs ViNeoBERT" throughout `evaluation_code/`.

## How NeoBERT / SALT3 uses this
PhoBERT plays **two roles**: (1) an **embedding donor** — the SALT init can recycle PhoBERT's
Vietnamese embeddings (the `phobert_procrustes_init_v6` / `phobert_*` arms in `compare-salt-versions.py`,
`salt3_diagnostics.py`); (2) the **headline downstream baseline** the adapted Vietnamese NeoBERT
("ViNeoBERT") is benchmarked against (all `0X_*_phobert_vs_vineobert` notebooks). Note the
**word-segmentation mismatch**: PhoBERT expects segmented input while NeoBERT/ViDeBERTa do not, which
matters when mining anchors and mapping donor rows. → [[salt]], [[procrustes]], [[videberta]].

## Relation: [[roberta]] [[videberta]] [[salt]] [[procrustes]] [[neobert]] [[culturax]]
