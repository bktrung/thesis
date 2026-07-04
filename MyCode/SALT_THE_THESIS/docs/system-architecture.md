# System Architecture: Vietnamese NeoBERT via SALT

## High-Level Pipeline

The pipeline is divided into three sequential stages, each producing reusable artifacts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Stage 1: SALT Initialization                       │
│                   01_init_embeddings_salt3.ipynb                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Input:                                                                   │
│  • NeoBERT (English encoder) from HuggingFace                            │
│  • ViDeBERTa (Vietnamese DeBERTa) from HuggingFace                       │
│  • Vietnamese FastText embeddings (cc.vi.300.bin)                        │
│                                                                           │
│  Pipeline:                                                                │
│  1. Load & validate NeoBERT and ViDeBERTa                               │
│  2. Prune ViDeBERTa tokenizer (128K → 32K)                              │
│  3. Mine anchor pairs (3-tier: numbers + verified forms + translations) │
│  4. Download FastText (if needed)                                        │
│  5. SALT projection: per-token least-squares mapping                    │
│  6. Norm calibration: match NeoBERT distribution                        │
│  7. Save init artifact + verify round-trip fidelity                     │
│                                                                           │
│  Output:                                                                  │
│  ✓ init/<INIT_NAME>/model/                                              │
│    ├── config.json (NeoBERT architecture, vocab_size=32K)               │
│    ├── model.safetensors (projected embeddings + decoder)               │
│    ├── tokenizer.json (pruned ViDeBERTa, 32K tokens)                    │
│    ├── special_tokens_map.json                                          │
│    ├── model.py (NeoBERT custom code)                                   │
│    ├── rotary.py (RoPE implementation)                                  │
│    └── salt_config.json (metadata: source, target, anchor count)        │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stage 2: Continual Pre-Training (CPT)                 │
│                      02_train_cpt_run.ipynb                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Input:                                                                   │
│  • init/<INIT_NAME>/model/ (from Stage 1)                               │
│  • CulturaX-vi (Vietnamese subset of CulturaX, streamed from HF)        │
│                                                                           │
│  Pipeline:                                                                │
│  1. Load SALT-initialized model + pruned tokenizer                      │
│  2. Stream CulturaX-vi documents (configurable, default 3M docs)        │
│  3. Tokenize with pruned tokenizer                                      │
│  4. Cache to Arrow format (keyed by tokenizer SHA256)                   │
│  5. Create DataCollatorForLanguageModeling (20% MLM, 100% [MASK])      │
│  6. Build NeoMLMTrainer (handles NeoBERT's custom forward)             │
│  7. Train: eval every 200 steps, save every 200 steps, keep best 5     │
│  8. Early stopping on validation loss                                   │
│  9. Save final model to runs/<RUN_NAME>/final_model/                   │
│                                                                           │
│  Output:                                                                  │
│  ✓ runs/<RUN_NAME>/                                                     │
│    ├── final_model/ (HF model directory)                                │
│    ├── checkpoints/ (trainer intermediate checkpoints)                  │
│    ├── metrics.jsonl (streamed training metrics)                        │
│    ├── plots/ (loss curves)                                             │
│    └── run_config.json (run metadata)                                   │
│                                                                           │
│  ✓ datasets/ (cached Arrow, reused across runs)                         │
│    └── culturax_vi_{num_docs}_seq{max_seq_len}_tok{fingerprint}/       │
│        ├── train/                                                        │
│        └── validation/                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       Stage 3: Downstream Evaluation                     │
│                  03_evaluate_downstream_salt3.ipynb                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Input:                                                                   │
│  • runs/<RUN_NAME>/final_model/ (from Stage 2)                          │
│  • Downstream datasets: UIT-VSFC (sentiment), XNLI-vi (NLI)            │
│  • Baseline model: PhoBERT-base-v2 (for comparison)                     │
│                                                                           │
│  Pipeline:                                                                │
│  1. Load trained model                                                  │
│  2. For each downstream task:                                           │
│     a. Fine-tune NeoBERTSequenceClassifier on train set                │
│     b. Evaluate on test set (3 seeds)                                   │
│     c. Report macro F1 and accuracy                                     │
│  3. Compare against PhoBERT baseline                                     │
│  4. Generate comparison table                                            │
│                                                                           │
│  Output:                                                                  │
│  ✓ eval_results.json                                                     │
│    └── {task_name}: {macro_f1, accuracy, num_seeds, ...}               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Architecture

### 1. SALT Projection Engine (Stage 1)

#### Anchor Mining (3-Tier)

```
Tier 1: Shared Numbers
  Input: ViDeBERTa vocab + NeoBERT vocab
  Process: Find digit tokens in both
  Output: (vi_token="3", neo_token="3") pairs
  Count: ~10–20 pairs

Tier 2: Verified Shared Surface Forms
  Input: ViDeBERTa full words + NeoBERT full words (using tokenizer-specific heuristics)
  Process: 
    1. Extract all word-initial tokens from both (ViDeBERTa: ▁ prefix, NeoBERT: no ## prefix)
    2. Build candidate set = intersection of surface forms
    3. For each candidate, back-translate via MarianMT (vi→en): 
       - If translation is single English word AND roundtrip (en→vi) matches original
       - AND English word exists in NeoBERT vocab, anchor it
  Output: (vi_token, neo_token) pairs
  Count: ~100–500 pairs (depends on translation accuracy)

Tier 3: Systematic Translation Mining
  Input: Remaining single-word Vietnamese tokens (from ViDeBERTa)
  Process:
    1. Filter by strict Vietnamese phonotactics (onset/coda rules, tone marking rules)
    2. Try back-translation (Google + Marian) for each word
       - Tier 3a: If back-translation matches AND English output is single word → anchor
    3. For compound Vietnamese words (space-separated), try direct Google Translate
       - Tier 3b: If result is single English word in NeoBERT vocab → anchor
    4. Remaining single words: blacklist-filtered (common grammatical words), try translation
       - Tier 3c: Cap at 500 pairs
  Output: (vi_token, neo_token) pairs
  Count: ~200–1000 pairs (configurable by tier)
```

#### SALT Projection Algorithm

Per non-shared token `vi_token` in pruned ViDeBERTa:

```
1. FastText Similarity
   ft_vec = FastText(vi_token.replace('▁', ''))  // 300-dim Vietnamese
   ft_vec = normalize(ft_vec)
   
   for each anchor_token in anchors:
       ft_anchor = FastText(anchor_token.replace('▁', ''))
       sim[anchor] = cosine_sim(ft_vec, ft_anchor)

2. Sparsemax Neighborhood Selection
   sparse_support = Sparsemax(sim)  // Projects onto probability simplex
   nz_indices = indices where sparse_support > 0
   
   // Sparsemax is superior to top-k because it:
   // - Avoids overfitting to fixed k neighbors
   // - Dynamically selects support size
   // - Zero-probability anchors are truly excluded

3. Local Least-Squares Projection
   if |nz_indices| >= MIN_ANCHORS_FOR_LOCAL (8):
       // Build local mapping matrix from selected anchors
       E_t = [ViDeBERTa_embedding[anchor] for anchor in selected_anchors]  // (K, 768)
       E_s = [NeoBERT_embedding[anchor_pair] for anchor in selected_anchors]  // (K, 768)
       E_dec = [NeoBERT_decoder[anchor_pair] for anchor in selected_anchors]  // (K, 768)
       
       // Solve: E_t @ X_emb ≈ E_s  (least squares with L2 ridge)
       X_emb = (E_t^T @ E_t + ridge*I)^{-1} @ E_t^T @ E_s
       X_dec = (E_t^T @ E_t + ridge*I)^{-1} @ E_t^T @ E_dec
       
       // Project current token
       new_embedding = ViDeBERTa_embedding[vi_token] @ X_emb
       new_decoder_row = ViDeBERTa_embedding[vi_token] @ X_dec
   else:
       // Not enough anchors: random Gaussian init
       new_embedding ~ N(mean, std)  // Match NeoBERT distribution
```

#### Norm Calibration

After projection, non-anchor embeddings may have different scale than NeoBERT. Adjust:

```
scale = NeoBERT_mean_embedding_norm / SALT_non_anchor_mean_norm
if scale > 1.3:  // More than 30% difference
    scale_embeddings(non_anchor_mask, scale)
    scale_decoder_rows(non_anchor_mask, scale)
```

#### Re-injection & Verification

```
for each (vi_token, neo_token) in anchors:
    new_embedding[vi_id] = NeoBERT_embedding[neo_id]  // Exact copy
    new_decoder[vi_id] = NeoBERT_decoder[neo_id]

for each special_token (pad, mask, cls, sep, etc):
    new_embedding[special_id] = NeoBERT_embedding[special_id]  // Exact copy

Verify:
  for 10 random anchors:
      assert |loaded_emb[vi_id] - NeoBERT_emb[neo_id]| < 1e-6
  assert all special tokens match exactly
  assert forward_pass produces finite logits
```

---

### 2. Training Engine (Stage 2)

#### NeoMLMTrainer

NeoBERT's forward signature differs from HF standard. Custom trainer overrides:

```python
class NeoMLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # NeoBERT expects: input_ids, attention_mask
        # (NOT input_ids, attention_mask, token_type_ids, position_ids)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        labels = inputs.get("labels")  # (batch, seq_len), -100 for non-masked
        
        # MLM loss: cross_entropy ignoring -100 indices
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )
        return (loss, {"logits": logits}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, logits = self._neo_forward_with_loss(model, inputs)
        if prediction_loss_only:
            return loss, None, None
        return loss, logits, None
```

#### Data Pipeline

```
1. Stream CulturaX-vi from HuggingFace Datasets library
   • Configurable num_examples (default 3M)
   • Streaming = no local disk until tokenization

2. Tokenize with pruned ViDeBERTa tokenizer
   • Config: max_seq_len=1024, add_special_tokens=False
   • Chunk text into 1024-token chunks (carry-over between docs)

3. Cache to Arrow format
   • Cache key = (num_examples, max_seq_len, tokenizer_fingerprint)
   • Stored in `datasets/culturax_vi_{...}/`
   • Next run with same tokenizer: instant load from disk

4. Train/Eval split
   • 98% train, 2% validation (eval_ratio=0.02)
   • Seed=42 for reproducibility

5. Collate batches
   • DataCollatorForLanguageModeling
   • 20% MLM probability
   • 100% replacement with [MASK] (no 80/10/10 BERT mix)
   • Pad to batch multiple of 8
```

#### Training Loop

```
for epoch in range(1):  // Single epoch (num_train_epochs=1)
    for batch_idx, batch in enumerate(train_loader):
        // Forward + backward
        loss = compute_loss(model, batch)
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            
        if global_step % eval_steps == 0:
            eval_loss = evaluate(model, eval_loader)
            save_checkpoint(model, eval_loss)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break
        
        // Log metrics
        if global_step % logging_steps == 0:
            metrics = {
                "step": global_step,
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            log_metrics(metrics)
```

#### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| batch_size | 32 | Fits in GPU memory, sufficient for gradient signal |
| gradient_accum | 16 | Effective batch = 512 (common for BERT-scale models) |
| learning_rate | 1e-4 | Conservative for transfer learning (not 5e-5 cold-start) |
| warmup | 5% | Ramp up over first 5% of steps |
| scheduler | cosine | Smooth decay, no hard cutoff |
| eval_steps | 200 | Eval every ~100K tokens |
| save_steps | 200 | Checkpoint same frequency as eval |
| save_limit | 5 | Keep top 5 best checkpoints by eval_loss |
| mlm_probability | 0.20 | 20% of tokens masked |
| mask_replace_prob | 1.0 | 100% replaced with [MASK] (not BERT's 80/10/10) |
| max_steps | -1 | Train until end of epoch |

---

### 3. Evaluation Engine (Stage 3)

#### Downstream Task Setup

```
Task: UIT-VSFC (Vietnamese Sentiment)
  Dataset: train/test split
  Classes: positive, negative, neutral (3-way)
  Metric: macro F1, accuracy
  Fine-tune: NeoBERTSequenceClassifier on train, eval on test
  Seeds: 3 runs with different random inits

Task: XNLI-vi (Cross-lingual NLI)
  Dataset: Vietnamese subset of XNLI
  Classes: entailment, contradiction, neutral (3-way)
  Metric: macro F1, accuracy
  Fine-tune: NeoBERTSequenceClassifier
  Seeds: 3 runs
```

#### NeoBERTSequenceClassifier

```python
class NeoBERTSequenceClassifier(nn.Module):
    def __init__(self, base_model, num_labels, dropout=0.1):
        self.base_model = base_model
        self.dropout = Dropout(dropout)
        self.classifier = Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        // Get final hidden state
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]  // (batch, seq_len, hidden_size)
        
        // Mean pooling over sequence, respecting attention mask
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        
        // Classify
        logits = classifier(dropout(pooled))  // (batch, num_labels)
        
        // Loss if labels provided
        loss = None
        if labels is not None:
            loss = cross_entropy(logits, labels)
        
        return {"loss": loss, "logits": logits}
```

**Note**: Uses mean pooling (not [CLS] token) to be more robust to sequence length. Respects attention mask during pooling.

---

## Model Architecture Details

### NeoBERT (Source Model)

```
NeoBERT 28L × 768H × 12A
├── Embeddings (32K vocab → 768 dim, no token_type_ids)
├── 28 × EncoderBlock
│   ├── QKV projection (768 → 768×3, no bias)
│   ├── Multi-head self-attention with RoPE
│   │   └── Flash attention (xformers, if available)
│   ├── Output projection (768 → 768, no bias)
│   ├── Residual + dropout
│   ├── SwiGLU feedforward (768 → 3072 → 768)
│   └── Residual + dropout
├── Pre-RMSNorm (applied before output)
└── MLM Head: Linear(768 → vocab_size)
```

**Key properties**:
- **RoPE** (Rotary Position Embeddings): Encodes absolute position via rotation matrices
- **SwiGLU**: Gated linear unit activation, parameter-efficient
- **Pre-RMSNorm**: Layer norm applied *before* sublayer (vs. post-LN)
- **No token_type_ids**: Uses RoPE for position, discards token types

### ViDeBERTa (Donor for Vietnamese)

```
ViDeBERTa-base (86M params)
├── Embeddings (128K → 768 dim)
├── 12 × DeBERTaV3Block
│   ├── Relative position bias (disentangled)
│   ├── Multi-head attention
│   └── Feedforward
└── [Embeddings + decoder extracted for SALT]
```

Only the **embeddings** are used in SALT. The transformer layers are discarded; NeoBERT's encoder is kept.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Stage 1: SALT Initialization                      │
└─────────────────────────────────────────────────────────────────────────┘

   HF NeoBERT                    HF ViDeBERTa                   HF CulturaX-vi
         ↓                              ↓                              (unused)
         │                              │
         ├─→ Load & extract ←───────────┤
         │   embeddings,               │
         │   decoder,                  │
         │   special tokens            │
         │                              ↓
         │                         Prune tokenizer
         │                         (128K → 32K)
         │                              ↓
         │                         Anchor mining
         │                          (3-tier mining)
         │                              ↓
         │                         FastText download
         │                              ↓
         │   ┌──────────────────────────┘
         │   │
         ├──→ SALT Projection
         │   (sparsemax + least-squares per token)
         │   ├→ Projected embeddings
         │   ├→ Projected decoder
         │   └→ Re-injected anchors + specials
         │
         ├──→ Norm calibration
         │
         └──→ Save init artifact
             ├── init/{INIT_NAME}/model/
             ├── config.json
             ├── tokenizer.json (pruned)
             └── model.safetensors

┌─────────────────────────────────────────────────────────────────────────┐
│                      Stage 2: Continual Pre-Training                     │
└─────────────────────────────────────────────────────────────────────────┘

  init/{INIT_NAME}/model/       HF CulturaX-vi
           ↓                           ↓
           │                    Stream documents
           │                           ↓
           │                    Tokenize (pruned tok)
           │                           ↓
           │                    Cache to Arrow
           │                           ↓
           ├──────────────────────────→ Collect batches
                                        ↓
                                   NeoMLMTrainer
                                   (20% MLM, 100% [MASK])
                                        ↓
                                   Training loop
                                   ├→ Forward pass
                                   ├→ Backward pass
                                   ├→ Optimizer step
                                   ├→ Eval every N steps
                                   ├→ Save checkpoints
                                   └→ Early stopping
                                        ↓
                                   Save final model
                                   └── runs/{RUN_NAME}/final_model/

┌─────────────────────────────────────────────────────────────────────────┐
│                     Stage 3: Downstream Evaluation                       │
└─────────────────────────────────────────────────────────────────────────┘

  runs/{RUN_NAME}/final_model/   UIT-VSFC + XNLI-vi
           ↓                              ↓
           │                        Load task dataset
           │                              ↓
           ├──→ Load model            Fine-tune on train
                                       (3 seeds)
                                              ↓
                                         Evaluate on test
                                              ↓
                                         Report metrics
```

---

## Key Design Decisions

### 1. Why Sparsemax Instead of Top-K?

- **Top-K**: Fixed neighbor set. If anchor is far from NeoBERT space, overfits to k neighbors.
- **Sparsemax**: Dynamically selects support. If token is isolated (low sim to all anchors), produces uniform distribution → falls back to random Gaussian.
- **Benefit**: Graceful degradation; avoids artificially constraining support size.

### 2. Why Local Least-Squares (Not Global)?

- **Global mapping** (one matrix for all tokens): Assumes one transformation suffices for all 32K tokens. Breaks for distant tokens.
- **Local mapping** (per-token using anchors): Uses anchor's local geometry. Sparsemax ensures only nearby anchors influence the transform.
- **Benefit**: Non-shared tokens project using neighbors' exemplars; distant tokens get random init.

### 3. Why 100% [MASK] Replacement (Not BERT's 80/10/10)?

- **BERT** (original): 80% mask, 10% random, 10% unchanged. Helps model learn to predict *any* word, not just [MASK].
- **NeoBERT pretraining** (design choice): Uses 100% [MASK] replacement. Model is pretrained on this objective.
- **CPT** (ours): Match pretraining objective for consistency. Using 80/10/10 would introduce distribution mismatch.
- **Empirical**: salt3_common.py comments note that random/unchanged significantly hurts results vs 100% [MASK].

### 4. Why Cache by Tokenizer Fingerprint?

- **Problem**: If tokenizer vocab changes (e.g., different pruning), cached Arrow dataset becomes invalid. Model would train on misaligned token IDs.
- **Solution**: Cache key includes `SHA256(tokenizer.vocab)`. Change vocab → new cache key → re-tokenize.
- **Benefit**: Prevents silent data corruption; enables safe tokenizer updates.

### 5. Why Arrow Format for Caching?

- **Arrow**: Memory-mapped, instant load, parallel reads, compressed storage.
- **Advantages**:
  - 3M docs tokenized once, reused across runs (saves hours)
  - 3M-doc cache ≈ 12 GB on disk
  - Loads in seconds vs. re-streaming + tokenizing (1 hour)
- **Trade-off**: Requires 12GB disk space, but worth it for repeated runs.

---

## Failure Modes & Diagnostics

### Failure Mode: Embedding Weights Not Loaded

**Symptom**: `eval_loss >> log(vocab_size)` (vocab_size ≈ 32K, log ≈ 10.4)

**Diagnosis** (via `diagnose_model_health()`):
- Embedding mean norm ≈ 0.02 (random init range, should be 0.3–0.5)
- Encoder weight std ≈ 0.02 (random init, should be varied)
- Forward pass entropy close to max entropy

**Cause**: `load_state_dict(..., strict=False)` with unexpected key mismatch

**Fix**: Verify safetensors integrity, check key names in model.py

---

### Failure Mode: Training Loss Diverges

**Symptom**: Loss NaN after 10–100 steps

**Diagnosis**:
- Check learning rate: 1e-4 standard; 1e-3 too high, 1e-5 too low
- Check gradient norm: should be stable, not exploding
- Check input logits entropy: should be reasonable, not uniform

**Causes**:
- Learning rate too high (use LR finder first)
- Attention heads producing NaN (xformers issue on old GPU)
- Invalid token IDs in batch (OOB token IDs)

**Fixes**:
- Lower learning rate to 5e-5
- Disable xformers: `model.config.use_cache = False`
- Verify dataset: `inspect_tokenized_dataset(dataset, tokenizer)`

---

### Failure Mode: Anchor Mining Produces Few Pairs

**Symptom**: Only 50 anchors when expecting 500+

**Causes**:
- Translation API rate limits (Google Translate)
- Tokenizer mismatch (ViDeBERTa vs NeoBERT word-initial detection)
- Overly strict Vietnamese phonotactics filter

**Fixes**:
- Increase translation batch size, add delays
- Verify `is_full_word_and_clean()` function against actual tokenizer output
- Relax blacklist for Tier 3

---

## Dependencies & Version Pinning

| Dependency | Min Version | Notes |
|---|---|---|
| torch | 2.5 | Flash attention, dtype handling |
| transformers | 4.46 | Custom model code support, API stability |
| xformers | 0.0.28 | Efficient attention (optional) |
| datasets | 2.x | HF Datasets, Arrow format |
| safetensors | 0.x | Safe model serialization |
| huggingface_hub | 0.x | Model/tokenizer download |
| fasttext-wheel | latest | FastText embeddings |
| deep_translator | latest | Google Translate API wrapper |

**Fragility**: torch + transformers + xformers are tightly coupled. Update all three together, or disable xformers if mismatch occurs.

---

## Performance Characteristics

| Operation | Time | Notes |
|---|---|---|
| Load NeoBERT + ViDeBERTa | ~10 min | HuggingFace Hub download + init |
| Anchor mining (1000 words) | ~1–2 hours | Includes Google Translate API calls, MarianMT batching |
| SALT projection (32K tokens) | ~30 min | GPU accelerated (CUDA) |
| Tokenize 3M docs | ~1 hour | First time; cached for future runs |
| CPT training (400K chunks, 1 epoch) | ~3–5 hours | On single GPU (A100 40GB) |
| Downstream eval (UIT-VSFC) | ~10 min | Fine-tune 3 seeds × ~3 min each |

**Optimization tips**:
- Use GPU for projection (sparsemax + LSQR on CUDA)
- Stream CulturaX-vi (don't download full dataset)
- Cache datasets aggressively (reuse across runs)
- Use bf16 precision if GPU supports (10–20% speedup)

