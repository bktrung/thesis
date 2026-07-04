# FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models

**Authors:** Konstantin Dobler, Gerard de Melo (Hasso Plattner Institute / University of Potsdam)
**Venue:** EMNLP 2023 Main Conference (Long Paper)
**arXiv:** 2305.14481
**Code:** https://github.com/konstantinjdobler/focus

## Abstract

The researchers address the challenge of adapting multilingual pretrained models to specific languages using new tokenizers. They propose "FOCUS - Fast Overlapping Token Combinations Using Sparsemax," describing it as "a novel embedding initialization method that initializes the embedding matrix effectively for a new tokenizer based on information in the source model's embedding matrix."

The approach represents newly added tokens as combinations of overlapping vocabulary tokens selected via semantic similarity in static embedding spaces. Using multilingual XLM-R as the source model, the method "outperforms random initialization and previous work in language modeling and on a range of downstream tasks (NLI, QA, and NER)."

## 1. Introduction

Multilingual language models show subpar performance on under-resourced languages. While crosslingual transfer using pretrained transformer weights is effective, direct embedding matrix transfer becomes impossible when introducing language-specific tokenizers. The paper argues that full vocabulary replacement with language-specific vocabularies offers practical advantages: XLM-R's 250k vocabulary reduced to 50k yields 55% parameter reduction and 40% faster training.

**Problem Statement:** How to initialize embeddings for a new target language tokenizer while preserving semantic information from source model pretraining?

## 2. FOCUS Method

### Overview

FOCUS uses three key components:

1. **Direct Transfer:** Copy embeddings for overlapping tokens between source and target vocabularies
2. **Auxiliary Embeddings:** Train fastText embeddings on target language text
3. **Weighted Initialization:** Initialize new tokens using weighted combinations of similar overlapping tokens

### Mathematical Formulation

Given:
- Source vocabulary V^s with pretrained embeddings E^s
- Target vocabulary V^t with embeddings E^t to initialize
- Overlap set O = V^s ∩ V^t

**Step 1 - Direct Transfer:**
```
For all o ∈ O: ê_o^t = ê_o^s
```

**Step 2 - Similarity Computation:**
For additional tokens A = V^t \ O, compute cosine similarities between target token auxiliary embeddings (X) and overlapping tokens:

```
sim(a,o) := (x_a · x_o) / (||x_a|| ||x_o||)
```

**Step 3 - Sparsemax Weighting:**
Convert similarities to sparse weights using sparsemax function:

```
w_a = sparsemax(c_a) = argmin_{p ∈ Δ} ||p - c_a||²
```

where Δ is the probability simplex.

**Step 4 - Weighted Initialization:**
```
For all a ∈ A: ê_a^t = Σ_{o∈O} w_{a,o} · ê_o^s
```

### Key Design Choices

**Sparsemax vs. Softmax:** Sparsemax produces exact zeros, dynamically accommodating varying sparsity in similarity distributions. This allows single highly-similar tokens to dominate initialization while multiple moderate-similarity tokens can contribute equally when appropriate.

**Auxiliary Embeddings:** FastText embeddings trained directly on target language text at token level outperform converting pretrained word-level embeddings, as demonstrated in experiments.

## 3. Experimental Setup

### Source Model & Tokenization

- **Source:** XLM-R (multilingual, 250k vocabulary)
- **Tokenization:** SentencePiece with Unigram algorithm
- **Target Vocabulary Size:** 50k tokens across all languages

### Baselines Compared

1. **XLM-R Original Vocabulary** (with/without language-adaptive pretraining [LAPT])
2. **Random Initialization** with optional embedding-only training for 20% of steps
3. **WECHSEL:** Bilingual embedding alignment using Procrustes + fastText (requires bilingual dictionaries)
4. **WECHSEL_En:** Restricted to English tokens in original vocabulary
5. **Vocabulary Extension:** Adding 30k target tokens to original 250k vocabulary with random or FOCUS initialization

### Language-Adaptive Pretraining (LAPT)

Configuration across all languages:
- **Objective:** Masked Language Modeling (MLM)
- **Data Source:** CC100 corpus
- **Training Scale:** 50 million samples (12.8 billion tokens, 390k optimizer steps)
- **Batch Size:** 128 sequences × 256 tokens
- **Learning Rate:** 5×10⁻⁵ (constant schedule with 5M sample warmup)
- **Optimizer:** AdamW

### Evaluation Tasks & Datasets

**Downstream Task Languages:** German (high-resource), Arabic & Kiswahili (medium), Hausa & isiXhosa (low-resource)

**Tasks & Datasets:**
- **NLI:** XNLI (translated, translate-train setting)
- **QA:** GermanQuAD (German), TyDiQA GoldP (Arabic, Swahili)
- **NER:** WikiANN, GermEval2014 (German), MasakhaNERv2 (Swahili, Hausa, isiXhosa)
- **MLM Evaluation:** Scottish Gaelic, Luxembourgish, Cebuano, Samoan, Hmong on mC4/OSCARv23.01

**Evaluation Protocol:** Five runs with different random seeds, reporting mean ± standard deviation. Model selection on dev split, reporting test results.

## 4. Results

### 4.1 Initialization Quality (Without Training)

**MLM Loss Immediately After Initialization:**

| Method | German | Arabic | Kiswahili | Avg Low-Resource |
|--------|--------|--------|-----------|------------------|
| Random | 24.0 | 24.1 | 24.2 | 23.2 avg |
| **FOCUS** | **4.0** | **5.2** | **4.8** | **6.1 avg** |
| FOCUS (Symbolic Only) | 10.6 | 10.6 | 10.7 | 9.2 avg |
| WECHSEL | 8.3 | 9.8 | 11.2 | — |

FOCUS dramatically outperforms all baselines, particularly on initialization without training.

### 4.2 Downstream Task Performance

**NLI and QA Results:**

| Task/Method | German | Arabic | Kiswahili | Average |
|-------------|--------|--------|-----------|---------|
| **XNLI (Replaced Vocab)** |
| Random + LAPT† | 77.6 ± 0.4 | 74.6 ± 0.4 | 71.2 ± 0.3 | 74.5 |
| WECHSEL + LAPT | 78.2 ± 0.2 | 76.0 ± 0.2 | 72.3 ± 0.3 | 75.5 |
| **FOCUS + LAPT** | **78.3 ± 0.6** | **76.5 ± 0.4** | **72.9 ± 0.5** | **75.9** |
| **GermanQuAD / TyDiQA** |
| Random + LAPT† | 69.1 ± 0.7 | 79.3 ± 0.6 | 74.2 ± 1.0 | 74.2 |
| WECHSEL + LAPT | 70.5 ± 0.5 | 79.4 ± 0.9 | 75.5 ± 1.5 | 75.1 |
| **FOCUS + LAPT** | **71.3 ± 0.2** | **79.1 ± 0.4** | **76.5 ± 1.5** | **75.6** |

**Named Entity Recognition (F1 Scores):**

| Method | German | Arabic | Kiswahili | Hausa | isiXhosa | Avg |
|--------|--------|--------|-----------|-------|----------|-----|
| Random + LAPT† | 86.0 | 87.5 | 85.8 | — | — | 86.4 |
| WECHSEL + LAPT | 86.5 | 87.9 | 87.4 | — | — | 87.3 |
| **FOCUS + LAPT** | **86.6** | **87.9** | **86.9** | — | — | **87.1** |

### 4.3 Key Findings

- FOCUS consistently outperforms random initialization across all downstream tasks
- Performance gains are most pronounced on low-resource languages (Hausa, isiXhosa)
- WECHSEL performs competitively but requires bilingual dictionaries; FOCUS doesn't
- Improvements persist across 50M-sample training regime despite initialization impact diminishing over extended training

### 4.4 Vocabulary Extension vs. Replacement

- **German (High-resource):** Original vocabulary remains strongest baseline
- **Arabic & Kiswahili:** Replacement with FOCUS > Extension > Original vocabulary
- **Practical Advantage:** Replacement achieves 55% parameter reduction and 40% faster training

### 4.5 Effect of Vocabulary Overlap

Testing with only numbers, punctuation, whitespace as overlap:

| Language | Full Overlap | Symbolic Only | Overlap Tokens |
|----------|--------------|---------------|----------------|
| German | 4.0 | 10.6 | 18,986 |
| Luxembourgish | 8.2 | 10.4 | ~12k |
| Cebuano (unseen) | 6.3 | 9.7 | ~11k |

FOCUS with symbolic-only overlap still outperforms WECHSEL on most languages.

## 5. Analysis & Discussion

### Vocabulary Overlap Quality

Manual analysis of 500 German overlapping tokens:

| Category | Share | Examples |
|----------|-------|----------|
| Symbols & Numbers | 9% | Numbers, punctuation, emojis |
| Names & Entities | 10% | Proper nouns, brand names |
| German Subwords | 46% | Native vocabulary, inflections |
| English & Code-switched | 18% | Borrowed terms, tech vocabulary |
| Unclassifiable | 17% | Noisy/rare tokens |

### Low-Resource Language Performance

FOCUS particularly benefits low-resource settings because:
1. No additional resources required (unlike WECHSEL's bilingual dictionaries)
2. Leverages target language tokens already in multilingual source model
3. Graceful degradation: works with minimal overlap through symbolic tokens
4. Requires only unlabeled target language text for fastText training

## 6. Related Work

### Embedding Initialization Methods

**Bilingual Alignment Approaches:**
- SMALA: Calculates cross-lingual embedding space mappings
- WECHSEL: Uses Procrustes method with bilingual dictionary seeds

FOCUS avoids these limitations by:
- Not requiring bilingual dictionaries
- Exploiting existing target language tokens in multilingual models
- Not assuming near-isomorphic embedding space structures

**Concurrent Work:** Ostendorff & Rehm (2023) propose similar approach using smaller pretrained transformer as auxiliary embedding space rather than fastText. FOCUS's fastText approach has lower computational cost.

## 7. Limitations

1. **Model Architecture:** Evaluation limited to BERT-like encoders; applicability to decoder models untested
2. **Language Coverage:** Downstream evaluations concentrate on German, Arabic, Swahili
3. **Data Availability:** Relies on monolingual text for LAPT; many truly low-resource languages lack sufficient data
4. **Script Coverage:** Limited evaluation on non-Latin script languages outside Arabic

## 8. Conclusion

FOCUS provides an effective, practical method for embedding initialization when specializing multilingual models with language-specific tokenizers:
- Outperforms random initialization and existing methods (WECHSEL) on downstream tasks
- Requires no additional resources beyond target language text
- Demonstrates robustness even with minimal vocabulary overlap
- Enables 55% model size reduction and 40% training speedup
- Particularly valuable for low-resource languages lacking bilingual dictionaries

**Key Innovation:** Sparsemax-weighted combinations of overlapping token embeddings as initialization, combined with directly trained token-level fastText embeddings for similarity computation.
