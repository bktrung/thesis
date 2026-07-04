# WECHSEL: Effective Initialization of Subword Embeddings for Cross-Lingual Transfer of Monolingual Language Models

**Authors:** Benjamin Minixhofer, Fabian Paischer, Navid Rekabsaz (Johannes Kepler University Linz, ELLIS Unit Linz, LIT AI Lab)
**Venue:** NAACL 2022
**arXiv:** 2112.06598
**Code:** https://github.com/CPJKU/wechsel

## Abstract

WECHSEL is a method for efficiently transferring pretrained monolingual language models to new languages. Rather than training from scratch, WECHSEL leverages multilingual static word embeddings to initialize subword token embeddings in target languages. The approach copies all non-embedding parameters from a source model (English RoBERTa or GPT-2) and replaces the tokenizer with one for the target language. The core innovation involves using aligned static word embeddings to compute semantically meaningful initializations for target language tokens by mapping them to similar source language tokens. WECHSEL-transferred models outperform those trained from scratch while requiring up to 64x less training effort, and also exceed performance of previously published monolingual models trained with substantially more resources.

## 1. Introduction

Large Transformer-based language models have become central to NLP applications, but training them requires enormous computational resources. The vast majority are trained on English, making extension to other languages prohibitively expensive. While massively multilingual models exist, research shows they suffer from a "curse of multilinguality" — performance degrades beyond a certain number of languages. Monolingual models typically outperform their multilingual counterparts, but training new monolingual models requires comparable computational investment to training the original English models.

Token embeddings constitute roughly 31% of RoBERTa and 33% of GPT-2 parameters. Rather than randomly initializing these embeddings during cross-lingual transfer, WECHSEL initializes them semantically by using multilingual static word embeddings.

## 2. Related Work

### Cross-lingual Transfer of Monolingual LMs

**Bilingualization** preserves source language capabilities while extending to target languages. Artetxe et al. (2020) replaced tokenizers and relearned embeddings while freezing other parameters. Tran (2020) used static word embeddings for semantic initialization but continued training on both languages.

**Monolingual creation** transfers knowledge without preserving source language performance. Earlier work (Zoph et al. 2016; Nguyen & Chiang 2017) used random token embeddings or vocabulary overlap. De Vries & Nissim (2021) introduced TransInner, training only embeddings initially before full model training.

WECHSEL belongs to the monolingual creation category, extending Tran's approach with arbitrary numbers of semantically similar subwords and flexible temperature parameters.

## 3. Methodology

### 3.1 Subword Embedding Computation

Given a tokenizer T with vocabulary U and static word embeddings W, the goal is computing subword embeddings U in the same semantic space as W.

**Process:** Decompose each subword into character n-grams and sum embeddings of occurring n-grams:

```
u_x = Σ(g ∈ G(x)) w_g
```

where G(x) is the n-gram set in subword x, and w_g is the n-gram embedding. This mirrors fastText's out-of-vocabulary word handling. Subwords containing no known n-grams initialize to zero.

### 3.2 Subword Similarity-based Transfer

After computing source (U^s) and target (U^t) subword embeddings:

1. **Compute cosine similarity** between every target subword and all source subwords:
```
s(x,y) = (u_x^t · u_y^s) / (||u_x^t|| ||u_y^s||)
```

2. **Initialize target embeddings** as weighted means of k-nearest source embeddings using softmax-weighted similarities with temperature τ:
```
e_x^t = Σ(y ∈ J_x) [exp(s(x,y)/τ) · e_y^s] / Σ(y' ∈ J_x) exp(s(x,y')/τ)
```

where J_x contains k nearest source language neighbors. Subwords with zero embeddings in U^t initialize from a normal distribution matching source embedding statistics.

**Hyperparameters:** τ=0.1 and k=10 selected through grid search using linear probes.

## 4. Experiment Design

### Models and Languages
- **RoBERTa** (125M parameters) transferred to French, German, Chinese, Swahili
- **GPT-2 small** (117M parameters) transferred to same languages plus low-resource (Sundanese, Scottish Gaelic, Uyghur, Malagasy)

### Evaluation
- **RoBERTa:** Fine-tuned on XNLI (NLI) and WikiANN (NER)
- **GPT-2:** Language Modeling Perplexity on held-out test sets

### Training Data
- Medium-resource: 4GiB subsets from OSCAR corpus
- Low-resource: CC-100 corpus (0.1-1.6GiB)

### Bilingual Embeddings
- fastText monolingual embeddings aligned using Orthogonal Procrustes
- Bilingual dictionaries from MUSE (French, German, Chinese), FreeDict (Swahili), Wiktionary (low-resource)

### Baselines
1. **FullRand:** Training from scratch in target language
2. **TransInner:** Random embedding initialization, copying non-embedding parameters, two-phase training

### Training Setup
- 250k steps on TPUv3-8 (~4 days per model)
- Identical hyperparameters across languages

## 5. Results

### 5.1 RoBERTa Transfer Results

**NLI (XNLI) Performance:**
- French: WECHSEL improves 7.15% absolute accuracy over FullRand
- German: +6.31% over FullRand
- Chinese: +6.94% over FullRand
- Swahili: +4.71% over FullRand

Compared to published monolingual models:
- French NLI: 1.55% improvement over CamemBERT
- German NLI: 3.15% improvement over GBERTBase

**Training Efficiency:**
- French: WECHSEL outperforms CamemBERT after 10% of steps (64x reduction in training effort)
- German: Outperforms GBERTBase after 10% of steps (39x reduction)

### 5.2 GPT-2 Transfer Results

#### Medium-Resource Languages

| Language | WECHSEL PPL | TransInner PPL | FullRand PPL |
|----------|-------------|----------------|--------------|
| French   | 19.71       | 20.13          | 20.47        |
| German   | 26.80       | 27.76          | 27.63        |
| Chinese  | 51.97       | 56.17          | 52.98        |
| Swahili  | 10.14       | 10.28          | 10.58        |

#### Low-Resource Languages

| Language | WECHSEL PPL | TransInner PPL | FullRand PPL |
|----------|-------------|----------------|--------------|
| Sundanese | 111.72     | 151.86         | 149.46       |
| Scottish Gaelic | 16.43 | 18.62         | 19.53        |
| Uyghur | 34.33        | 39.06          | 42.82        |
| Malagasy | 14.01       | 14.85          | 15.93        |

### 5.3 Freezing Analysis

WECHSEL performs well without parameter freezing, while TransInner requires freezing to prevent catastrophic forgetting — indicating WECHSEL's superior initialization quality.

## 6. Limitations and Risks

- Experiments limited to 8 languages
- Extrinsic evaluation covers only two tasks (NLI, NER)
- English-trained LMs encode societal biases; transferred models likely inherit these

## 7. Conclusion

WECHSEL enables efficient transfer of monolingual language models to new languages by leveraging multilingual static word embeddings for semantically informed subword initialization. Key results:
- Transferred models outperform both training-from-scratch baselines and published monolingual models
- Up to 64x reduction in training effort
- Supports the hypothesis that deep monolingual models learn cross-linguistically generalizable abstractions

## Key Contributions

1. Novel parameter transfer method using multilingual embeddings for subword initialization
2. Successful transfer of RoBERTa and GPT-2 to multiple medium- and low-resource languages
3. Up to 64x reduction in training effort vs published baselines
4. Publicly released code and models
