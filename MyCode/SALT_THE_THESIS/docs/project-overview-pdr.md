# Vietnamese NeoBERT via SALT: Product Development Requirements

## Executive Summary

This thesis project adapts **NeoBERT** (250M-param English encoder, SOTA benchmark performer) to Vietnamese using **SALT (Semantic Aware Linear Transfer)**, a novel cross-lingual transfer method. Instead of pretraining from scratch (expensive), we transfer ViDeBERTa's (Vietnamese DeBERTa) tokenizer and embeddings into NeoBERT's embedding space, then run continual pre-training (CPT) on Vietnamese text. Current status: prototype running on Google Colab.

---

## Functional Requirements

### 1. SALT Embedding Initialization Pipeline (Stage 1)
- Load NeoBERT (source encoder) and ViDeBERTa (target PLM for Vietnamese embeddings)
- Prune ViDeBERTa's 128K tokenizer down to ~32K (match NeoBERT vocab size)
- Mine anchor pairs via 3-tier translation: shared surface forms + back-translation verification + systematic translation mining
- Download Vietnamese FastText embeddings (cc.vi.300.bin)
- Apply SALT projection per token: sparsemax neighborhood selection → local least-squares linear transform from ViDeBERTa embedding space to NeoBERT space
- Apply norm calibration to match NeoBERT's embedding distribution
- Save reusable init artifact: model + projected embeddings + new decoder + pruned tokenizer
- Verify round-trip fidelity, special token exactness, forward pass

### 2. Continual Pre-Training (Stage 2)
- Load SALT-initialized model from init artifact
- Stream and tokenize CulturaX-vi dataset (Vietnamese, configurable up to 3M docs)
- Implement MLM training loop with custom NeoMLMTrainer (handles NeoBERT's non-standard forward signature)
- Support checkpoint resume and continuation runs
- Log metrics to JSONL, plot training curves
- Hyperparameters: batch 32, grad_accum 16, lr 1e-4, cosine schedule, 20% MLM, early stopping

### 3. Downstream Evaluation (Stage 3)
- UIT-VSFC: Vietnamese sentiment classification (3-class)
- XNLI-vi: Cross-lingual NLI (3-class)
- Multi-seed evaluation with baseline comparisons (PhoBERT-base-v2)
- Custom NeoBERTSequenceClassifier wrapper using mean pooling

---

## Non-Functional Requirements

| Requirement | Target |
|---|---|
| **Scalability** | Codebase must support 3M-doc CPT runs on single GPU (batching, caching strategy) |
| **Reproducibility** | Tokenizer fingerprinting, seed control, metrics logging |
| **Fragility** | Minimize HF remote code dependency; document breaking changes |
| **Code quality** | Modular design, clear error messages, schema validation |
| **Data integrity** | Cached datasets keyed by tokenizer SHA256; no stale data reuse |

---

## Key Models

### NeoBERT (Source Architecture)
- **HuggingFace**: `chandar-lab/NeoBERT`
- **Paper**: arXiv:2502.19587
- **Params**: 250M (28 layers × 768 hidden, 12 attention heads)
- **Architecture**: SwiGLU activation, RoPE positional embeddings, Pre-RMSNorm
- **Context length**: 4096 tokens
- **Pretraining**: RefinedWeb (2.8TB), 2.1T tokens, 20% MLM (100% [MASK] replacement)
- **Vocab**: WordPiece, 30,522 + specials = 32,064
- **Fragility**: Uses `trust_remote_code=True` (downloads `model.py`, `rotary.py` at runtime)

### ViDeBERTa (Donor PLM)
- **HuggingFace**: `Fsoft-AIC/videberta-base`
- **Paper**: ViDeBERTa: Vietnamese DeBERTaV3
- **Params**: 86M (base)
- **Tokenizer**: SentencePiece, 128K vocab (pruned to 32K in SALT init)
- **Pretraining**: CC100 Vietnamese (138GB)
- **Architecture**: DeBERTaV3

### SALT Method
- **Paper**: arXiv:2505.10945v2
- **Core idea**: Transfer embeddings from target-language PLM (ViDeBERTa) to source LLM (NeoBERT) via sparse linear transforms
- **Steps**: FastText subword embeddings → anchor mining → sparsemax neighborhood → per-token least-squares projection → norm calibration → CPT
- **Anchor mining**: Shared numbers + verified surface forms + 3-tier translation (back-translation + compound + blacklist-filtered single words)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: SALT Init (01_init_embeddings_salt3.ipynb)             │
├─────────────────────────────────────────────────────────────────┤
│  Load NeoBERT + ViDeBERTa                                       │
│  Prune ViDeBERTa tokenizer 128K → 32K                           │
│  Mine anchors (surface forms + translations)                    │
│  Download FastText embeddings (cc.vi.300.bin)                   │
│  SALT projection per token → new embeddings + decoder           │
│  Norm calibration                                               │
│  Save: init/<INIT_NAME>/model/                                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: CPT Training (02_train_cpt_run.ipynb)                  │
├─────────────────────────────────────────────────────────────────┤
│  Load SALT-initialized model                                    │
│  Stream & tokenize CulturaX-vi                                  │
│  MLM training loop (NeoMLMTrainer)                              │
│  Checkpointing, resume, continuation                            │
│  Save: runs/<RUN_NAME>/final_model/                             │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Evaluation (03_evaluate_downstream_salt3.ipynb)        │
├─────────────────────────────────────────────────────────────────┤
│  Load trained model                                             │
│  UIT-VSFC (sentiment), XNLI-vi (NLI)                            │
│  Multi-seed eval, baseline comparisons                          │
│  Report metrics                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

1. **Anchor Mining**: Surface forms (both tokenizers) + MarianMT back-translation + Google Translate systematic mining
2. **FastText Projection**: 300-dim Vietnamese FastText → 768-dim NeoBERT space via local least-squares + sparse neighborhood
3. **Dataset**: CulturaX-vi (HF streamed) → tokenized (pruned ViDeBERTa tokenizer) → cached Arrow format (keyed by tokenizer SHA256)
4. **Training**: Cached dataset → DataCollatorForLanguageModeling (100% [MASK] replacement) → NeoMLMTrainer → checkpoints → final model

---

## Known Risks & Fragilities

### 1. HuggingFace Remote Code Dependency
- NeoBERT model code (`model.py`, `rotary.py`) downloaded from HF Hub at `trust_remote_code=True`
- **Risk**: If HF changes the model card or repository, pipeline breaks
- **Mitigation**: Vendor model code locally in a future iteration

### 2. Library Version Coupling
- Tight coupling: HF transformers + xformers (SwiGLU) + custom model code
- **Risk**: Updates to `transformers` or `xformers` can break forward pass
- **Current status**: Known breakage due to recent library changes (to be fixed)

### 3. salt3_common.py Monolith
- **Current**: 771 LOC of utilities + model wrappers + training infrastructure in one file
- **Risk**: Difficult to maintain, unclear boundaries
- **Mitigation**: Split into modular components once prototype is stable

### 4. Translation Mining Brittleness
- 3-tier mining depends on Google Translate API (rate limits, availability)
- MarianMT may produce incorrect translations for low-freq or OOV words
- **Mitigation**: Cache all translation results; provide fallback anchor sets

### 5. Embedding Norm Calibration
- Norm scaling heuristic (when SALT mean norm < 70% of NeoBERT) may not generalize to all datasets
- **Mitigation**: Monitor embedding norms during CPT; adjust scaling if test loss diverges

---

## Current Project Phase

**Phase**: Prototype / Thesis Proof-of-Concept
- Running on Google Colab with Google Drive storage
- Manual run orchestration via Jupyter notebooks
- Known breakage in library stack (to be diagnosed and fixed)

**Next priorities**:
1. Fix current library breakage
2. Vendor HF model code locally
3. Modularize salt3_common.py
4. Scale CPT run to full 3M-doc CulturaX-vi
5. Run comprehensive downstream evaluation suite

---

## Success Criteria

- ✓ Init artifact builds without errors, embedding round-trip fidelity < 1e-5
- ✓ CPT training loop runs without NaNs, loss decreases monotonically
- ✓ Downstream eval: XNLI-vi and UIT-VSFC accuracy better than random baseline
- ✓ CPT perplexity on Vietnamese text lower than cold-start PhoBERT
- (Stretch) Competitive with monolingual Vietnamese BERT variants on benchmark suite

---

## References

- NeoBERT paper: arXiv:2502.19587
- SALT paper: arXiv:2505.10945v2
- ViDeBERTa paper: (included in `papers/` directory)
- Papers in full-text Markdown: `papers/neobert.md`, `papers/salt.md`, `papers/videberta.md`
