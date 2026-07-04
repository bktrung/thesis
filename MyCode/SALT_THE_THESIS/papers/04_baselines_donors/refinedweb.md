---
title: "The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only"
authors: Penedo, Malartic, Hesslow, Cojocaru, Cappelli, Alobeidli, Pannier, Almazrouei, Launay
year: 2023
venue: NeurIPS 2023 (Datasets & Benchmarks)
arxiv: 2306.01116
url: https://arxiv.org/abs/2306.01116
tags: [dataset, pretraining-corpus, web-data, neobert-data, english]
---

# RefinedWeb — the English pre-training corpus NeoBERT is built on

## Core idea
Shows that **properly filtered and deduplicated web data alone** (CommonCrawl only, no curated
books/Wikipedia mix) can match or beat models trained on curated corpora. RefinedWeb is the
resulting large, clean English web corpus (a ~600B-token public extract of a ~5T-token pipeline),
originally built to train the **Falcon** LLMs.

## Key details — the MDR pipeline
- **Macrodata Refinement (MDR):** aggressive **document- and line-level filtering** (language ID,
  quality heuristics, adult/boilerplate removal) applied to raw CommonCrawl WARC.
- **Two-stage deduplication:** fuzzy (MinHash) + exact substring dedup — deduplication is identified
  as the single biggest driver of quality.
- Thesis: quantity of *clean, deduplicated* web tokens > curated-corpus purity; curation is
  substitutable by scale + rigorous filtering.

## Results / why it matters
Falcon models trained purely on RefinedWeb were competitive with models trained on curated data,
validating "web-only, but well-refined" pre-training. It became a standard high-quality English
pre-training source.

## How NeoBERT / SALT3 uses this
**NeoBERT is pre-trained on RefinedWeb** — this is the corpus that produced the frozen English body
SALT3 adapts ([[neobert]]). It matters to the thesis for two reasons: (1) it defines the *English
representation space* the Vietnamese embeddings are transferred **into** (so the geometry SALT3
aligns to was learned from English web text); (2) it is the English analogue of the Vietnamese CPT
corpus [[culturax]] — the language-adaptation step swaps RefinedWeb-English for CulturaX-Vietnamese
while reusing the same body. Cited in the thesis's NeoBERT background where its pre-training data is
described.

## Relation: [[neobert]] [[culturax]] [[bert]] [[roberta]]
