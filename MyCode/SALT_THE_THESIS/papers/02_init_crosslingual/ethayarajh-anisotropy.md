---
title: "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"
authors: Ethayarajh
year: 2019
venue: EMNLP 2019
arxiv: 1909.00512
url: https://arxiv.org/abs/1909.00512
tags: [theory, geometry, anisotropy, init-justification]
---

# Embedding geometry / anisotropy (Ethayarajh)

## Core idea
The embedding spaces of BERT/ELMo/GPT-2 are **anisotropic**: vectors occupy a **narrow cone**, not a uniform sphere. So two random words already have high cosine similarity, and the geometry is *locally* structured rather than globally uniform. This is the key geometric fact that justifies **local, per-token** cross-space maps over one global linear transform.

## Key findings / details
- **Anisotropy:** word vectors cluster in a narrow cone; cosine similarity is inflated (any two words look somewhat similar). Worse in upper layers.
- **Context-specificity:** upper layers produce more context-specific representations; self-similarity of a word across contexts drops with depth.
- **<5% of variance:** less than 5% of the variance in a word's contextualized representations is explained by a single static vector — i.e. one vector per token cannot summarize its behavior.
- Implication: a **single global linear map** between two anisotropic spaces cannot capture their locally-varying relationships; the right tool is a map that **adapts per token / per neighborhood**.

## Why it matters
Provides the geometric reason that cross-lingual/cross-model embedding transfer should be *local*, and explains why cosine-based anchor selection must be used carefully (everything is somewhat similar in an anisotropic space → need a sparse, selective rule like sparsemax).

## How NeoBERT / SALT3 uses this
This is a **theoretical pillar of the thesis's design choice**: SALT3 fits a **per-token local weighted-least-squares map** (one `X` per token from its sparsemax-selected anchors), *not* a single global donor→NeoBERT matrix. Anisotropy ([[ormazabal-mapping-limits]] adds non-isomorphism) is the reason a global map underfits and the local maps win — the justification to write in Ch.2/Ch.3 for "why per-token." It also motivates **sparsemax** anchor selection ([[sparsemax]]): in an anisotropic space a dense softmax over anchors blurs everything, so a sparse support is needed.

## Relation: [[ormazabal-mapping-limits]] [[representation-degeneration]] [[salt]] [[sparsemax]] [[procrustes]]
