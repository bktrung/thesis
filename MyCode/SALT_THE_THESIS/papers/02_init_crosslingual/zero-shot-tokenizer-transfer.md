---
title: Zero-Shot Tokenizer Transfer
authors: Minixhofer, Ponti, Vulić
year: 2024
venue: NeurIPS 2024
arxiv: 2405.07883
url: https://arxiv.org/abs/2405.07883
tags: [init, crosslingual, context]
---

# Zero-Shot Tokenizer Transfer (ZeTT)

## Core idea
Detach a language model from its tokenizer. Train a **hypernetwork** that takes *any* tokenizer as
input and **predicts the embedding (and LM-head) matrix** for that tokenizer's vocabulary — so you
can swap tokenizers "on the fly" with near-zero performance loss, instead of using a heuristic init
that "performs at chance level." This is the most general / learned end of the embedding-init
spectrum that WECHSEL→FOCUS→SALT occupy with closed-form heuristics.

## Key math / architecture details
- **Hypernetwork** `H(tokenizer) → (E, W_head)`: conditions on the token *strings/byte sequences*
  to generalize to unseen vocabularies, rather than fitting one fixed new matrix.
- Trained by sampling many tokenizers and teaching `H` to reproduce good embeddings; generalizes to
  both encoders (XLM-R) and decoder LLMs (Mistral-7B).
- Heuristic inits (WECHSEL/FOCUS/SALT) are *training-free* and use auxiliary static spaces or anchor
  overlaps; ZeTT instead *amortizes* the init into a learned predictor.
- **< 1B tokens** of continued training closes any residual gap — directly motivating a short CPT
  phase after init.

## Results / why it matters
Near-original accuracy after tokenizer swap on cross-lingual and coding tasks, with shorter
sequences. Positions tokenizer/vocabulary transfer as a first-class capability and frames the
"init then briefly continue-train" recipe SALT3 follows.

## How NeoBERT / SALT3 uses this
ZeTT is **context, not a component** SALT3 implements. It matters two ways: (1) it confirms the
thesis's overall recipe — *replace the tokenizer + init embeddings well + short continued
pre-training* — is the right frame, and quantifies the "< 1B tokens to close the gap" budget that
informs SALT3's CPT horizon (CulturaX-vi up to a few-M-doc ceiling); (2) it is the natural
"learned-init" comparison point to SALT3's **closed-form, anchor-based** init, useful for the
related-work discussion of why a heuristic transfer is chosen here (no hypernetwork training, single
target language).

## Relation: [[wechsel]] [[focus]] [[salt]] [[dont-stop-pretraining]]
