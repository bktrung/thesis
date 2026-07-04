# OFA: A Framework of Initializing Unseen Subword Embeddings for Efficient Large-scale Multilingual Continued Pretraining

**Authors:** Yihong Liu, Peiqin Lin, Mingyang Wang, Hinrich Schütze (LMU Munich / Munich Center for Machine Learning)
**Venue:** Findings of NAACL 2024
**arXiv:** 2311.08849
**Code:** https://github.com/cisnlp/ofa

## Abstract

OFA (One For All) is a framework designed to efficiently extend pretrained language models to new languages through smart initialization of subword embeddings combined with matrix factorization. Rather than randomly initializing embeddings for new subwords during vocabulary expansion, OFA leverages external well-aligned multilingual static word vectors (ColexNet+ covering 1,335 languages) and injects alignment knowledge into subword embeddings.

## Core Innovation: Embedding Factorization

The framework factorizes source embedding matrix E^s ∈ R^(|V^s|×D) into:
- F^s ∈ R^(|V^s|×D') (language-specific coordinates)
- P ∈ R^(D'×D) (language-agnostic primitive embeddings)

This reduces trainable parameters from |V^s|×D to |V^s|×D' + D'×D.

## Methodology

### OFA Framework Steps

**Step 1:** Factorize source embeddings using SVD, selecting top-k eigenvalues

**Step 2:** Create bipartite graph between external vocabulary and source subwords, initializing subword embeddings as averages of connected word embeddings

**Step 3:** Repeat process for target vocabulary

**Step 4:** Initialize target coordinates by:
- Copying overlapping subwords from source coordinates
- Computing cosine similarity between source/target embeddings
- Initializing non-overlapping subwords as weighted combinations of k-nearest source coordinates

**Step 5:** Assemble target model with transformer body, primitive embeddings, and target coordinates

## Experimental Setup

### Models Tested
- **OFA-mono-xxx:** RoBERTa (English) extended to multilingual
- **OFA-multi-xxx:** XLM-R extended to more languages
- **Baselines:** RoBERTa-rand, XLM-R-rand (random initialization)

### Latent Dimensions
Tested D' = 100, 200, 400, 768 (no factorization)

### Downstream Tasks

**Sentence Retrieval:**
- Bible (SR-B): up to 500 English-aligned sentences, 275 tail languages
- Tatoeba (SR-T): up to 1,000 sentences, 70 head languages

**Sequence Labeling:**
- NER (WikiANN): 7 classes, F1 score
- POS (Universal Dependencies): 18 classes, F1 score

**Text Classification:**
- Taxi1500: 6 classes across 354 languages, F1 score
- Zero-shot evaluation (English training, multilingual testing)

### Training Configuration
- Corpus: Glot500-c (511 languages, 1.5B sentences)
- Batch size: 384 (effective)
- Learning rate: 5e-5
- Sequence length: 512
- FP16 training
- Hardware: 4 NVIDIA RTX A6000 GPUs
- Maximum duration: 4 weeks

## Key Results

### Performance Comparison

**OFA-mono-768 vs RoBERTa-rand:**
- SR-B: +6.0 improvement
- SR-T: +6.6 improvement
- Taxi1500: +8.3 improvement
- NER: +4.3 improvement
- POS: +5.1 improvement

**OFA-multi-768 vs XLM-R-rand:** Consistent improvements across all tasks.

### Parameter Efficiency

When D'=100:
- Model parameters: 126M (vs 395M at full dimension)
- Embedding parameters: 40M (vs 309M)
- Training time per 10K steps: 8.4 hours
- Carbon footprint: 47.9 kg CO₂ eq.

### Convergence Analysis

OFA models demonstrate significantly faster convergence than random baselines. Lower-dimensional models (D'=200-400) often achieve better downstream performance than full-dimensional versions.

## Critical Observations

**Dimension-Performance Trade-off:**
- Consistent improvement from D'=100 to D'=400
- Improvement plateaus or decreases at D'=768
- Suggests redundancy in multilingual embeddings (confirmed by PCA analysis)

**Task-Specific Patterns:**
- Sequence labeling shows minimal improvement with continued pretraining
- Retrieval and classification tasks benefit substantially
- Syntactic knowledge transfers rapidly; semantic knowledge requires more training

**Environmental Impact:**
Lower-dimensional OFA variants achieve equivalent or superior performance with substantially reduced training time, GPU consumption, and carbon emissions (up to 59% reduction).

## Ablation Studies

### Continued Pretraining Effect

Performance without continued pretraining:
- Monolingual source: Poor across all dimensions
- Multilingual source: Acceptable for some tasks, especially head languages
- OFA-multi-100 surpasses OFA-multi-768 after 10K training steps on most tasks

### Redundancy Analysis

PCA dimension reduction shows:
- Monolingual models (BERT, GPT-2): ~30% variance at 100 components
- Multilingual models (XLM-R, Glot500-m): 40-50% variance at 100 components
- Higher redundancy correlates with better cross-lingual transfer ability

## Limitations

1. Only tested on encoder-only models (RoBERTa, XLM-R), not decoder-only or encoder-decoder
2. Only MLM-based pretraining tested
3. Potential catastrophic forgetting: external knowledge injected into embeddings may diminish during continued pretraining

## Conclusion

OFA provides a practical, efficient framework for multilingual language model adaptation by combining informed embedding initialization from external multilingual resources with parameter-efficient factorization. The framework achieves better performance with less GPU consumption, making it particularly valuable for resource-constrained settings.

## Comparison with Other Methods

| Method | Auxiliary Source | Dictionary? | Factorization? | Languages Covered |
|--------|----------------|-------------|----------------|-------------------|
| WECHSEL | fastText aligned | Yes | No | Per-pair |
| FOCUS | fastText trained | No | No | Per-language |
| CLP-Transfer | Small PLM | No | No | Per-language |
| **OFA** | **ColexNet+** | **No** | **Yes (SVD)** | **1,335 languages** |
| SALT | Target PLM | No | No | Per-language |
