---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: Devlin, Chang, Lee, Toutanova
year: 2019
venue: NAACL 2019
arxiv: 1810.04805
url: https://arxiv.org/abs/1810.04805
tags: [architecture, foundation, mlm]
---

# BERT

## Core idea
Pre-train a **bidirectional** Transformer encoder with a self-supervised **Masked Language
Model (MLM)** objective, then fine-tune the whole model on downstream tasks. Bidirectionality
(seeing left *and* right context jointly) is the key departure from left-to-right LMs and is
what makes BERT a strong *representation* model rather than a generator.

## Key math / architecture details
- **MLM:** randomly mask a fraction of input tokens; predict them from the full bidirectional
  context with a softmax over the vocabulary. Original recipe masks **15%** of tokens; of those,
  80% → `[MASK]`, 10% → random token, 10% → unchanged (to reduce train/test mismatch since
  `[MASK]` never appears at fine-tuning).
- **NSP (Next Sentence Prediction):** binary classification of whether sentence B follows A.
  Later shown by RoBERTa to be unnecessary/harmful; NeoBERT drops it.
- **Architecture:** Transformer encoder, learned absolute position embeddings, GELU activation,
  LayerNorm (post-norm), WordPiece tokenizer. BERT-base = 12 layers / 768 hidden / 12 heads
  (110M); BERT-large = 24 / 1024 / 16 (340M).
- **Input:** `[CLS] sentence_A [SEP] sentence_B [SEP]` with segment + position embeddings added
  to token embeddings. `[CLS]` pooled representation used for sentence-level tasks.

## Results / why it matters
SOTA on GLUE, SQuAD, SWAG on release; established the **pre-train-then-fine-tune** paradigm for
encoders. Every model in `04_baselines_donors/` (RoBERTa, ModernBERT, PhoBERT, ViDeBERTa) is a
BERT-lineage encoder.

## How NeoBERT / SALT3 uses this
NeoBERT keeps BERT's **MLM pre-training objective** but modernizes everything around it:
RoPE instead of learned positions, RMSNorm/pre-norm, SwiGLU, no NSP, and a higher
**20% masking rate** (vs BERT's 15%) — see the repo `NeoBERT/README.md` feature table.
SALT3's continued pre-training (CPT) on Vietnamese is itself MLM training: the
`salt3_*` code masks tokens and computes cross-entropy against the (re-initialized) LM head,
directly continuing BERT's objective in a new language.

## Relation: [[transformer]] [[roberta]] [[neobert]] [[modernbert]]
