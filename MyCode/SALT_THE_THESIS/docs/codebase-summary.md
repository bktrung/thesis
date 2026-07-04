# Codebase Summary: Vietnamese NeoBERT via SALT

## Directory Structure

```
SALT_THE_THESIS/
├── code/                          # Main pipeline notebooks & shared utilities
│   ├── salt3_common.py            # (771 LOC) Shared utilities library
│   ├── 00_colab_runbook.ipynb     # Setup checklist and folder contract
│   ├── 01_init_embeddings_salt3.ipynb  # SALT init pipeline
│   ├── 02_train_cpt_run.ipynb     # Continual pre-training
│   ├── 03_evaluate_downstream_salt3.ipynb  # Downstream task eval
│   ├── 16_staged_wsd_cpt.ipynb    # Staged WSD CPT (resumable, windowed, milestones)
│   ├── salt3_staged_schedule.py   # WSD LR (callback-driven) + cooldown branch
│   ├── salt3_staged_cpt.py        # run_session orchestrator + atomic rolling save
│   ├── salt3_staged_cpt_manifest.py    # per-arm two-level-resume manifest
│   ├── salt3_staged_cpt_callbacks.py   # milestone snapshot + global-step metrics
│   ├── salt3_staged_cpt_plot.py        # budget-sliceable curves
│   ├── salt3_staged_cpt_checks.py      # CPU smoke tests (torch-free Tier A)
│   ├── 20_context_extension_4k.ipynb   # 4k stage-2 finetune + PPPL-vs-length figure
│   ├── salt3_ctx4096.py                # seq-4096 window planner + regroup + trainer (manifest-neutral)
│   ├── salt3_pppl_curve.py             # pseudo-perplexity vs length, two-panel before/after figure
│   └── salt3_ctx4096_checks.py         # CPU smoke tests (31, torch/datasets-free)
├── NeoBERT/                       # Git submodule: NeoBERT official repo
│   ├── src/neobert/model/
│   │   ├── model.py               # (651 LOC) Core NeoBERT architecture
│   │   ├── rotary.py              # RoPE positional embeddings
│   │   └── rmsnorm.py             # RMSNorm implementation
│   ├── conf/                      # Hydra configs for pretraining
│   ├── scripts/                   # Pretraining and evaluation scripts
│   ├── pyproject.toml             # Pinned deps: torch 2.5, transformers 4.46
│   └── README.md                  # NeoBERT usage instructions
├── papers/                        # Full-text Markdown versions of papers
│   ├── salt.md                    # SALT paper (arXiv:2505.10945v2)
│   ├── neobert.md                 # NeoBERT paper (arXiv:2502.19587)
│   └── videberta.md               # ViDeBERTa paper
├── docs/                          # Project documentation
│   ├── project-overview-pdr.md    # PDR and project overview
│   ├── codebase-summary.md        # This file
│   ├── code-standards.md          # Code style and conventions
│   ├── system-architecture.md     # Architecture and data flow
│   └── project-roadmap.md         # Implementation phases and milestones
└── plans/                         # Implementation plans and reports
```

---

## Core Files

### salt3_common.py (771 LOC)
**Purpose**: Centralized utilities library for all three stages of the pipeline.

**Modules**:

| Function Group | Lines | Purpose |
|---|---|---|
| **Google Colab integration** | 18–40 | Drive mounting, path resolution for Colab vs local |
| **File I/O utilities** | 42–85 | Path creation, JSON/JSONL read/write, JSON append |
| **Seed & environment** | 18–24, 48–54 | Set random seeds, configure HF remote code trust |
| **Model loading** | 195–238 | `load_model_safe()` with graceful missing-key handling, safetensors/bin fallback |
| **Embedding extraction** | 251–268 | Robust weight matrix extraction from arbitrary model architectures |
| **Fingerprinting** | 270–312 | Embedding norm stats, tokenizer vocab hash (SHA256) for cache keys |
| **Dataset utilities** | 315–328 | Tokenized dataset sanity checks |
| **Training infrastructure** | 380–648 | `NeoMLMTrainer` (custom loss + pred_step), collators, arg builders, checkpoint logic, metrics logging |
| **Evaluation wrappers** | 409–457 | `NeoBERTSequenceClassifier`, `NeoBERTQuestionAnswering` |
| **Model diagnostics** | 662–770 | `diagnose_model_health()` — embedding norms, encoder weight stats, forward pass sanity, mask probe |

**Key classes**:
- **NeoMLMTrainer**: Trainer subclass overriding `compute_loss()` and `prediction_step()` to handle NeoBERT's custom forward signature (uses `input_ids` + `attention_mask`, not the standard transformers API)
- **NeoBERTSequenceClassifier**: Mean-pooling classifier wrapper (respects attention mask)
- **JsonlMetricsCallback**: Callback to stream training metrics to JSONL file

**Dependencies**: torch, transformers, datasets, huggingface_hub, safetensors, numpy, fasttext (optional), deep_translator (optional)

---

### Staged WSD CPT harness (`16_staged_wsd_cpt.ipynb` + `salt3_staged_*.py`)
**Purpose**: Single-arm, config-driven continued-pretraining that reuses ONE tokenized "ceiling"
cache across inits, trains arbitrary per-session doc budgets, resumes after crashes, advances to NEW
data windows on continuation (never re-reads doc 0), logs one budget-sliceable metrics file, captures
milestone snapshots, and runs a WSD learning-rate schedule that never re-warms on resume.

**Modules** (each < 200 LOC; thin notebook = orchestration only):

| File | Role |
|---|---|
| `salt3_staged_schedule.py` | Pure WSD LR (`wsd_lr_at`/`compute_warmup_steps`/`cooldown_lr_at`, torch-free), LR-control callbacks (overwrite `param_groups['lr']` each step → version-independent), optimizer-state carry, `run_cooldown_branch` (cosine→0 fork; trunk untouched) |
| `salt3_staged_cpt.py` | `run_session` — windows the next chunk slice, two-level resume, advances manifest after a clean return, milestone snapshots, table-milestone cooldowns; atomic rolling `current_model` save + `.bak` recovery |
| `salt3_staged_cpt_manifest.py` | Per-arm `manifest.json` I/O (global_step/chunks/docs/milestones), atomic write, advance-once |
| `salt3_staged_cpt_callbacks.py` | Milestone stable-snapshot (global-step crossing, idempotent) + continuous global-step metrics (budget fields + true applied LR) |
| `salt3_staged_cpt_plot.py` | `plot_by_budget` — train/eval loss + ppl vs docs/tokens, slice-to-budget |
| `salt3_staged_cpt_checks.py` | CPU smoke harness: Tier A torch-free (21 checks), Tier B Colab-only |

**Reuses** `salt3_common`: additive windowing on `make_mlm_datasets` (`chunk_start`/`chunk_end`/`eval_chunks`), `cache_meta.json` + `read_cache_meta`, `assert_shared_tokenizer`, `docs_to_chunks`/`chunks_to_steps`. Legacy callers (nb02/nb15) are byte-identical (no new kwargs → legacy front-slice path).

---

### 4k context extension (`20_context_extension_4k.ipynb` + `salt3_ctx4096.py` / `salt3_pppl_curve.py`)
**Purpose**: NeoBERT stage-2 analog — fork the parked `milestone_5000000_decay` snapshot and continue at seq **4096** over a fresh unseen CulturaX window at constant LR 1e-5, then reproduce NeoBERT Figure 2 (pseudo-perplexity vs sequence length, before/after). Manifest-neutral (mirrors `salt3_fresh_decay`), single-A100 session.

| File | Role |
|---|---|
| `salt3_ctx4096.py` | `plan_ctx4096_window` (pure planner: ~200M-token budget → steps, clamped single-pass to the unseen 1024-tail), `regroup_to_4096` (4 consecutive seq-1024 chunks → one seq-4096 row; relies on sequential per-split packing), `run_ctx4096_extend` (read-only fork, `make_wsd_lr_callback` warmup→constant, fresh AdamW β2=0.95, resumable `save_steps`, writes `milestone_5000000_ctx4096/`) |
| `salt3_pppl_curve.py` | Subsampled pseudo-perplexity (`PPPL = exp(mean CE)`, 128 independently-masked positions/seq) on long VN-Wikipedia docs; `run_pppl_figure` loads both checkpoints via `load_model_safe(...AutoModelForMaskedLM)`, draws two-panel before/after figure + JSONL |
| `salt3_ctx4096_checks.py` | CPU smoke harness (31 checks, torch/datasets-free via AST-extracted pure fns): planner single-pass/no-repeat/clamp, byte-exact regroup, PPPL masking + math, WSD LR plateau |

**Reuses** `salt3_common` (`make_mlm_datasets` windowing, `read_cache_meta`, `load_model_safe`, `make_neo_mlm_trainer_class`, `training_args`) and `salt3_staged_schedule` (`build_wsd_optimizer`, `make_wsd_lr_callback`, `build_mlm_collator_compat`). No new scheduler, no re-streaming.

---

### 00_colab_runbook.ipynb
**Purpose**: Interactive setup checklist and environment validation.

**Cells**:
- Mount Google Drive
- Verify folder structure contract: `{PROJECT_ROOT}/init/`, `{PROJECT_ROOT}/runs/`, `{PROJECT_ROOT}/datasets/`
- Validate HF token and model access
- Print environment info (GPU, CUDA, transformers version)

---

### 01_init_embeddings_salt3.ipynb
**Purpose**: SALT embedding initialization pipeline (Stage 1).

**Pipeline**:

| Phase | Input | Output |
|---|---|---|
| **Load & verify** | NeoBERT + ViDeBERTa from HF | Model configs, embedding shapes, special token IDs |
| **Prune tokenizer** | ViDeBERTa 128K vocab | Pruned tokenizer (32K), mapping tables |
| **Anchor mining** | Pruned + NeoBERT vocabs, FastText | Anchor pairs (vi_token → neo_token) via 3 tiers |
| **SALT projection** | Target embeddings + anchor pairs + FastText | Projected embeddings (32K × 768) + new decoder |
| **Norm calibration** | Projected embeddings | Scaled embeddings (matching NeoBERT distribution) |
| **Save artifact** | Model + embeddings + tokenizer | `/init/<INIT_NAME>/model/` + config JSON |
| **Verify** | Saved checkpoint | Reload test, round-trip error < 1e-5 |

**Outputs** (under `/content/drive/MyDrive/SALT3/init/<INIT_NAME>/`):
- `model/`: HF-compatible model directory (config.json, model.safetensors, tokenizer.json, special_tokens_map.json, model.py, rotary.py)
- `salt_config.json`: Init metadata (source/target models, vocab size, anchor count, decoder init strategy)
- `init_embedding_fingerprint.json`: Embedding norm stats for verification
- `salt_anchor_pairs.csv`: All anchor mappings
- `cc.vi.300.bin`: Vietnamese FastText embeddings (downloaded once, cached)

---

### 02_train_cpt_run.ipynb
**Purpose**: Continual pre-training on Vietnamese text (Stage 2).

**Pipeline**:

| Phase | Action |
|---|---|
| **Config** | `MODE` (new/resume/continue), `RUN_NAME`, `BASE_MODEL_REF` (points to init artifact), learning rate, batch size, etc. |
| **Load init** | Load model from `BASE_MODEL_REF` via `load_model_safe()` |
| **Dataset** | Stream CulturaX-vi, tokenize with pruned tokenizer, cache to Arrow (keyed by tokenizer SHA256) |
| **Training setup** | Build TrainingArguments, NeoMLMTrainer, metrics callback |
| **Train loop** | MLM objective, checkpoint every N steps, evaluate on validation set, early stopping |
| **Save** | Final model to `runs/<RUN_NAME>/final_model/` |

**Hyperparameters** (defaults in `salt3_common.training_args()`):
- Batch size: 32 per device × 16 grad_accum = effective 512
- Learning rate: 1e-4, cosine decay with 5% warmup
- MLM probability: 20%, mask replacement: 100% (no random/unchanged)
- Eval: every 200 steps, save every 200 steps, keep best 5 checkpoints
- Precision: bf16 if available, else fp16

**Outputs**:
- `runs/<RUN_NAME>/final_model/`: Trained model
- `runs/<RUN_NAME>/metrics.jsonl`: Training metrics (loss, eval_loss, perplexity per step)
- `runs/<RUN_NAME>/plots/`: Training curves (matplotlib PNG)

---

### 03_evaluate_downstream_salt3.ipynb
**Purpose**: Downstream task evaluation (Stage 3).

**Tasks (10, GLUE-type-aligned)**: PPPL; UIT-VSFC (sentiment), UIT-VSMEC (emotion),
UIT-ViCTSD (toxicity); XNLI-vi (translated NLI) + ViANLI (native NLI); STS (semantic
similarity, regression/Spearman, `GreenNode/stsbenchmark-sts-vn`); QNLI (derived from
`taidng/UIT-ViQuAD2.0`); UD-VTB POS; MRC (UIT-ViQuAD2.0, full data).

**Models (4)**: SALT3 (NeoBERT heads) vs PhoBERT-v2, ViDeBERTa-base (donor), XLM-R base.

**Unified runner** (`eval_seq_task`) — one protocol for all: hybrid HP grid (small tasks)
or tuned LR (big), true early stopping, 5 seeds, VnCoreNLP segmentation for PhoBERT/ViDeBERTa.
Grid runs are isolated/unrecorded; only final best-LR per-seed runs write `seed_{s}/` artifacts
+ `RESULTS_JSONL`. Stats: mean±std + bootstrap 95% CI + paired-bootstrap significance vs SALT3
(in `salt3_common.py`); NeoBERT task-type coverage table. CoLA gap + MTEB are documented future work.
5. Report macro F1, accuracy

**Outputs**: Eval results JSON per task, comparison table

---

## NeoBERT Submodule

Located at `NeoBERT/` (git submodule, reference only).

### model.py (651 LOC)
- **NeoBERTConfig**: Config dataclass (28 layers, 768 hidden, 12 heads, 4096 context, SwiGLU, RoPE, RMSNorm)
- **EncoderBlock**: Single transformer block (QKV projection, attention, feedforward with SwiGLU)
- **NeoBERT**: Full encoder (embeddings + stack of encoder blocks + pre-RMSNorm)
- **NeoBERTForMaskedLM**: Adds MLM head (logits projection)
- **NeoBERTForSequenceClassification**: Adds sequence classification head

### rotary.py
- `precompute_freqs_cis()`: Pre-compute RoPE frequencies
- `apply_rotary_emb()`: Apply rotations to query/key matrices

### rmsnorm.py
- **RMSNorm**: Root mean square layer normalization (alternative to LayerNorm)

---

## Folder Structure & Data Contract

### Google Drive Layout (Colab)
```
/content/drive/MyDrive/SALT3/
├── code/
│   ├── salt3_common.py
│   └── *.ipynb
├── init/
│   └── {INIT_NAME}/
│       ├── model/              # HF model artifact
│       ├── salt_config.json
│       └── init_embedding_fingerprint.json
├── runs/
│   └── {RUN_NAME}/
│       ├── final_model/        # Trained model
│       ├── checkpoints/        # Training checkpoints
│       ├── metrics.jsonl
│       ├── plots/
│       └── run_config.json
└── datasets/
    └── culturax_vi_{num_docs}_seq{max_seq_len}_tok{fingerprint}/
        ├── train/             # Arrow parquets
        └── validation/
```

### Local Layout (Development)
Same structure under current working directory if `/content/drive/MyDrive` doesn't exist.

---

## Module Dependencies

```
01_init_embeddings_salt3.ipynb
├── salt3_common
├── transformers (AutoTokenizer, AutoModel, AutoModelForMaskedLM)
├── datasets (load_dataset, stream CulturaX-vi)
├── fasttext
├── deep_translator (GoogleTranslator)
├── Helsinki-NLP/opus-mt-vi-en (MarianMT)
└── pandas, numpy, torch

02_train_cpt_run.ipynb
├── salt3_common
├── transformers (Trainer, TrainingArguments)
├── datasets (load cached Arrow)
└── torch

03_evaluate_downstream_salt3.ipynb
├── salt3_common
├── transformers
├── datasets
└── sklearn.metrics
```

---

## Key Design Patterns

### 1. Artifact-Based Workflow
- **Init artifacts** (`init/<INIT_NAME>/model/`) are immutable, reusable snapshots
- **Run artifacts** (`runs/<RUN_NAME>/`) are training outputs, keyed by run name
- Enables multiple CPT runs to share the same init artifact without re-tokenizing

### 2. Tokenizer Fingerprinting
Dataset caches are keyed by `tokenizer_fingerprint()` (SHA256 of vocab + special token IDs + sample encodings). Prevents silent reuse of stale data if tokenizer changes.

### 3. Trust Remote Code
All models use `trust_remote_code=True` to load custom architectures from HF Hub. Salt3_common handles this transparently.

### 4. Custom Trainer Override
`NeoMLMTrainer` overrides Trainer's `compute_loss()` and `prediction_step()` to adapt to NeoBERT's forward signature (which differs from HF standard).

### 5. Graceful Model Loading
`load_model_safe()` handles missing/unexpected keys, tries both safetensors and bin formats, and prints detailed diagnostic info.

---

## Known Limitations & TODOs

| Issue | Impact | Mitigation |
|---|---|---|
| 771-line monolith (salt3_common.py) | Hard to navigate | Modularize once prototype stable |
| HF remote code download at runtime | Fragile to HF changes | Vendor model code locally |
| Google Translate in anchor mining | Rate-limited, availability risk | Cache all translations, provide fallback anchors |
| Manual Colab orchestration | Not scalable | Automate via scripts or job scheduler |
| Limited downstream eval suite | Unknown generalization | Add more tasks (Vietnamese SQuAD, semantic similarity) |

---

## Version & Dependency Notes

**pinned** (from NeoBERT/pyproject.toml):
- torch >= 2.5
- transformers >= 4.46
- xformers >= 0.0.28

**optional**:
- fasttext-wheel (for FastText embeddings)
- deep_translator (for Google Translate)
- datasets (streaming + Arrow caching)
- pandas (anchor mining, evaluation)
- matplotlib (plotting)
- scikit-learn (metrics)

---

## Testing & Validation

### salt3_common Functions
- `diagnose_model_health()`: Full model diagnostics (embedding norms, encoder weight stats, forward pass, mask probe)
- `embedding_fingerprint()`: Embedding integrity check
- `tokenizer_fingerprint()`: Vocab and special tokens validation
- `inspect_tokenized_dataset()`: Dataset sanity checks (ID bounds, unknown tokens)

### Notebook-Level Checks
- Init notebook: Embedding round-trip fidelity < 1e-5, anchor spot checks, forward pass no NaN
- CPT notebook: No divergence in loss, gradient norms stable, checkpoints recoverable
- Eval notebook: Task metrics reproducible across seeds, baselines valid

---

## Entry Points

**For new users**:
1. Read `00_colab_runbook.ipynb` (setup checklist)
2. Run `01_init_embeddings_salt3.ipynb` with default config (creates init artifact)
3. Run `02_train_cpt_run.ipynb` with small dataset (400K docs) for sanity check
4. Run `03_evaluate_downstream_salt3.ipynb` to see downstream performance

**For developers**:
1. Understand `salt3_common.py` entry points: `load_model_safe()`, `make_neo_mlm_trainer_class()`, `make_mlm_datasets()`
2. Review `01_init_*.ipynb` SALT projection logic (sparsemax, least-squares)
3. Review `02_train_*.ipynb` trainer setup and metrics callback
