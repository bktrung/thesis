---
title: "OFA: A Framework of Initializing Unseen Subword Embeddings for Efficient Large-scale Multilingual Continued Pretraining"
authors: Liu, Lin, Wang, Schütze
year: 2024
venue: NAACL 2024 Findings
arxiv: 2311.08849
url: https://arxiv.org/abs/2311.08849
tags: [init, crosslingual, baseline]
---

# OFA (embedding-init baseline)

## Core idea
Initialize the embeddings of an expanded multilingual vocabulary by **grounding new subwords in well-aligned external multilingual static word vectors**, then **factorize** the embedding matrix into two low-rank matrices to cut parameters. Like WECHSEL/FOCUS it replaces random init with a similarity-grounded one, but adds a factorization for large-scale multilingual continued pretraining.

## Key details
- **External alignment:** uses aligned multilingual static vectors (a shared crosslingual space) to place unseen subword embeddings — same "similarity in an auxiliary space" idea as WECHSEL/FOCUS ([[wechsel]], [[focus]]).
- **Matrix factorization:** `E ≈ U V` (low-rank), reducing the parameters of a huge multilingual embedding table and sharing structure across languages.
- **Goal:** adapt one PLM to *many* languages efficiently; **accelerates CPT convergence** vs default (random-init) continued pretraining.

## Results / why it matters
Competitive-or-better than default CPT baselines across crosslingual downstream tasks, with faster convergence and lower compute/carbon. A standard modern point of comparison in the "smart embedding init for vocabulary adaptation" family.

## How NeoBERT / SALT3 uses this
OFA is a **related-work baseline** in the WECHSEL → FOCUS → OFA → SALT → SALT3 lineage of embedding-initialization methods (Ch.2). It is the *multilingual, factorized* branch; SALT3 differs by (a) targeting **one** language (Vietnamese) with a **specific donor PLM** (ViDeBERTa) rather than generic multilingual static vectors, (b) using **per-token local WLS maps** with mined translation-pair anchors, and (c) initializing an **untied decoder** + freq-bias. OFA's "accelerates CPT convergence vs random init" claim is the same headline benefit SALT3 argues, supporting the thesis's motivation. → [[salt]], [[wechsel]], [[focus]].

## Relation: [[wechsel]] [[focus]] [[salt]] [[dont-stop-pretraining]]
