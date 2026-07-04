---
title: "FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models"
authors: Dobler, de Melo
year: 2023
venue: EMNLP 2023 (Main)
arxiv: 2305.14481
url: https://arxiv.org/abs/2305.14481
tags: [init, crosslingual, sparsemax, lineage]
---

# FOCUS (direct ancestor of SALT3's sparsemax average)

## Core idea
When specializing a multilingual model (e.g. XLM-R) to one language with a new tokenizer, split the
target vocabulary into **overlapping** tokens (present in both tokenizers) and **new** tokens.
**Copy** overlapping-token embeddings directly from the source model; initialize each **new** token
as a **sparsemax-weighted combination of the overlapping tokens'** embeddings, weighted by semantic
similarity in an auxiliary fastText space. Sparsemax makes the combination **sparse and
interpretable** (only a few donors per new token).

## Key math / architecture details
- **Overlap copy:** `E_target[t] = E_source[t]` for `t ∈ overlap` (anchors).
- **New tokens:** train fastText on target text; embed each target token; for a new token `n`,
  compute similarities to the *overlapping* tokens, then
  `E_target[n] = Σ_{o∈overlap} sparsemax(sim(n,o))_o · E_source[o]`.
- **Sparsemax** (→ [[sparsemax]]) zeroes out all but the most similar anchors → each new embedding
  is a convex combo of a handful of semantically-close, *already-aligned* overlap tokens.
- No bilingual dictionary needed (unlike WECHSEL): the overlap tokens are the bridge.

## Results / why it matters
Beats random init and WECHSEL on language modeling and downstream (NLI, QA, NER) for low-resource
specialization. Crucially introduces **sparsemax over anchor tokens** as the combination rule — the
exact mechanism SALT3 reuses.

## How NeoBERT / SALT3 uses this
FOCUS is the **direct ancestor of SALT3's embedding construction**. SALT3 generalizes FOCUS:
- **Anchors** are not just *string-overlap* tokens but **mined cross-lingual pairs** (3-tier:
  verified translations, shared numbers/symbols, similarity-mined) — see `salt_anchor_pairs.csv`,
  `read_anchor_map()` in `salt3_decoder_variants.py`.
- The new embedding is a **sparsemax-weighted average of donor (ViDeBERTa/PhoBERT) rows mapped into
  NeoBERT space**, i.e. `sparsemax(ft @ anchor_ftᵀ)` over anchor donors — FOCUS's rule applied
  across *two different models* rather than within one.
- SALT3 adds a **decoder/LM-head** construction and **frequency bias** ([[freq-bias-init]]) that
  FOCUS does not address.

## Relation: [[wechsel]] [[salt]] [[sparsemax]] [[freq-bias-init]]
