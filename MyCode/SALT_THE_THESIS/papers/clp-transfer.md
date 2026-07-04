# CLP-Transfer: Efficient Language Model Training through Cross-Lingual and Progressive Transfer Learning

**Authors:** Malte Ostendorff, Georg Rehm (DFKI GmbH, Berlin)
**Venue:** arXiv 2023
**arXiv:** 2301.09626
**Code:** https://github.com/malteos/clp-transfer

## Abstract

CLP-Transfer is a methodology for resource-efficient training of large language models targeting underrepresented languages. Rather than training from scratch, the approach transfers knowledge from pretrained source language models (like English) combined with smaller target language models. The technique can reduce training requirements by up to 80% compared to random initialization.

## 1. Introduction

Most Transformer-based language models are predominantly pretrained on English text, creating performance disparities for other languages. CLP-Transfer extends prior cross-lingual transfer work by incorporating model size progression (progressive transfer). Instead of directly training a large target-language model, the method:
- Leverages a pretrained large source-language model
- Uses a smaller pretrained target-language model
- Initializes token embeddings from overlapping vocabularies
- Reuses remaining Transformer weights from source models

## 2. Related Work

### Cross-lingual Transfer
Prior approaches (Artetxe et al. 2020, de Vries & Nissim 2021) transferred models between languages by freezing most parameters and retraining only token embeddings. WECHSEL (Minixhofer et al. 2021) improved upon this by using bilingual dictionaries to map embeddings.

### Progressive Transfer
Progressive growth — gradually increasing model capacity — has been demonstrated across architectures. Gong et al. (2019) showed depth-based progressive BERT training achieved "25% shorter training time."

## 3. Methodology

### 3.1 Core Assumptions

**Shared Vocabulary Assumption:**
Source and target language tokenizers must share substantial vocabulary overlap:
- English GPT2 ↔ German: 24.04% overlap
- Multilingual BLOOM ↔ German: 5.55% overlap
- English GPT2 ↔ Finnish: 13.71% overlap

**Token Embeddings Assumption:**
Relative positioning in embedding space remains comparable across model sizes despite different hidden dimensions. Testing on OPT models showed 54% overlap in k=10 nearest neighbors between 125M and 13B models.

### 3.2 Technical Implementation

**Transformer Weight Initialization:**
```
W_t^(large) = W_s^(large)
```
All Transformer weights copied directly from source language large model.

**Token Embedding Initialization:**

For overlapping tokens (v ∈ V_s ∩ V_t):
```
v_t = v_s
```

For non-overlapping tokens (v ∉ V_s ∩ V_t):
```
v_t^(large) = Σ(v̂ ∈ V_s ∩ V_t) [v̂_s^(large) / δ(v_t, v̂_t)]
```

Weight function δ uses normalized cosine similarity of small model embeddings:
```
δ(v,v̂) = cos(v_t^(small), v̂_t^(small)) / Σ cos(v'_t^(small), v̂'_t^(small))
```

This preserves spatial relationships from the small target-language model while leveraging source-language knowledge.

## 4. Experiment Design

### 4.1 Models

**GPT2 Experiments:**
- Source: English GPT2-XL (1.5B params, 48 layers, 1600 hidden dim)
- Target: German GPT2-XL (1.5B params, same architecture)
- Small model: German GPT2-base (117M params, 12 layers, 768 hidden dim)

**BLOOM Experiments:**
- Source: Multilingual BLOOM 7.1B (30 layers, 4096 hidden dim)
- Target: German BLOOM 6.4B (30 layers, 4096 hidden dim)
- Small model: German BLOOM 1.5B (24 layers, 2048 hidden dim)

### 4.2 Evaluation Tasks

- Language modeling: OSCAR validation perplexity
- Zero-shot downstream: GermEval 2017 (sentiment), GermEval 2018 (hate speech), GNAD10 (topic classification), PAWSX (paraphrase), XNLI (inference), X-Stance (stance detection)

### 4.3 Baselines

1. **From-Scratch Training:** Random weight initialization
2. **WECHSEL:** Bilingual dictionary-based cross-lingual transfer
3. **Multilingual Models:** XGLM (564M-7.5B), mGPT (1.3B)

## 5. Results

### 5.1 GPT2 Transfer (1.5B Parameters)

| Method | Final PPL | Tokens to Match From-Scratch |
|--------|-----------|------------------------------|
| CLP-Transfer | 12.8 | 15.4B (50% of training) |
| WECHSEL | 13.5 | — |
| From-Scratch | 15.1 | 30.8B |

CLP-Transfer achieved equivalent performance consuming only half the tokens.

### 5.2 BLOOM Transfer (6.4B Parameters)

| Method | Final PPL | Tokens to Match From-Scratch |
|--------|-----------|------------------------------|
| CLP-Transfer | 44.1 | ~10B (20% of training) |
| From-Scratch | 69.3 | 50.4B |

CLP-Transfer achieved superior results with 80% fewer training tokens.

### 5.3 Downstream Task Performance

Zero-shot evaluation across six German benchmarks showed most models achieving near-random results — attributed to insufficient model scale and token count.

## 6. Conclusion

CLP-Transfer successfully reduces training effort for large language models targeting underrepresented languages by:
- Combining cross-lingual and progressive transfer
- Exploiting overlapping vocabularies
- Preserving embedding space properties
- Achieving 50-80% training efficiency gains

**Key Innovation:** Using a smaller pretrained target-language model as the auxiliary embedding space (instead of fastText), enabling progressive transfer across both languages AND model sizes simultaneously.

## Comparison with Other Methods

| Method | Aux Embeddings | Bilingual Dict? | Progressive? | Max Savings |
|--------|---------------|-----------------|--------------|-------------|
| WECHSEL | fastText | Yes | No | 64x steps |
| FOCUS | fastText | No | No | ~40% speed |
| CLP-Transfer | Small PLM | No | Yes | 80% tokens |
| SALT | Target PLM | No | No | Fastest convergence |
