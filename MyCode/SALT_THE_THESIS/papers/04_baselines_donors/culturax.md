---
title: "CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages"
authors: Nguyen, Nguyen, Lai, Man, Ngo, Dernoncourt, Rossi, Nguyen
year: 2023
venue: LREC-COLING 2024
arxiv: 2309.09400
url: https://arxiv.org/abs/2309.09400
tags: [dataset, vietnamese, cpt-corpus]
---

# CulturaX (the CPT corpus)

## Core idea
A massive, **cleaned and deduplicated** multilingual corpus — 6.3T tokens across **167 languages** —
combining mC4 and OSCAR with a rigorous pipeline (language ID, multi-stage cleaning, fuzzy
dedup, perplexity/heuristic filtering). Provides high-quality, ready-to-train **Vietnamese** text at
scale, which is the practical enabler of monolingual continued pre-training.

## Key details
- Built from mC4 + OSCAR; aggressive cleaning + MinHash near-dedup; per-language quality filtering.
- Released fully (HuggingFace), making the Vietnamese split easy to stream/tokenize and cache.
- **Unsegmented** raw text — matches NeoBERT/ViDeBERTa subword tokenization (no VnCoreNLP word
  segmentation needed, unlike PhoBERT-style pipelines → [[phobert]]).

## Results / why it matters
One of the largest open, cleaned multilingual corpora; the default choice for adding a language to a
model via continued pretraining.

## How NeoBERT / SALT3 uses this
**CulturaX-vi is SALT3's continued-pretraining corpus.** The thesis streams and tokenizes the
Vietnamese split through the pruned tokenizer and caches it to Drive (the
`culturax_vi_{N}_seq1024_*` caches referenced throughout `code/`), uses it to **count the Vietnamese
unigram log-frequencies** for the decoder bias ([[freq-bias-init]]), and trains the WSD CPT on it
(`19_fresh_data_decay_5m`, `salt3_fresh_decay.py` use *fresh, non-recycled* CulturaX chunks for the
decay phase, per Hägele → [[hagele-cooldown]]). The repo notes CulturaX is "unsegmented, which is
exactly what CPT will [see]" — i.e. a clean match to the model's tokenizer. → [[dont-stop-pretraining]],
[[wsd-minicpm]].

## Relation: [[dont-stop-pretraining]] [[wsd-minicpm]] [[hagele-cooldown]] [[freq-bias-init]] [[neobert]]
