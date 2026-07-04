---
title: "Nomic Embed: Training a Reproducible Long Context Text Embedder"
authors: Nussbaum, Morris, Duderstadt, Mulyar
year: 2024
venue: TMLR 2024
arxiv: 2402.01613
url: https://arxiv.org/abs/2402.01613
tags: [baseline, encoder, embeddings]
---

# NomicBERT / Nomic Embed (nomic-bert-2048)

## Core idea
A fully **open and reproducible** long-context text embedding model. Train a modern BERT
(**nomic-bert-2048**) with a **2,048** context from scratch (MLM), then **contrastively fine-tune**
it into an embedding model (`nomic-embed-text-v1`) that beats OpenAI's `text-embedding-ada-002` and
`text-embedding-3-small` on MTEB — releasing **data, code, and weights**.

## Key math / architecture details
- **nomic-bert-2048 backbone:** modern encoder choices — RoPE positions ([[rope]]), SwiGLU FFN
  ([[swiglu]]), 2,048-token context — i.e. the same "LLM-ify the encoder" move as NeoBERT/ModernBERT.
- **Two stages:** (1) MLM pretraining of the backbone; (2) **contrastive** learning on large
  query–document pairs (in-batch negatives) to produce sentence embeddings.
- **Evaluation:** MTEB (short context) + LoCo (long context); emphasis on full reproducibility.

## Results / why it matters
Strong open embedding model; a key **MTEB baseline**. Demonstrates the now-standard recipe
*modern encoder → contrastive fine-tune → embeddings*, which is exactly the pipeline NeoBERT and an
adapted Vietnamese NeoBERT would slot into.

## How NeoBERT / SALT3 uses this
NomicBERT is one of the **baselines NeoBERT outperforms on MTEB**. For SALT3 it is the relevant
template for the **downstream use** of the adapted model: the thesis's evaluation suite
(`evaluation_code/`: retrieval, STS, reranking, clustering — MTEB-style, PhoBERT vs ViNeoBERT) mirrors
Nomic's contrastive-embedding evaluation. So NomicBERT both justifies the base-model choice and frames
how a Vietnamese NeoBERT would be turned into an embedding model. → [[neobert]].

## Relation: [[neobert]] [[modernbert]] [[rope]] [[swiglu]]
