# Code Standards: Vietnamese NeoBERT via SALT

## Overview

This document establishes coding conventions, style guidelines, and architectural patterns for the SALT_THE_THESIS project. The codebase is split between **production Python** (salt3_common.py) and **research Jupyter notebooks** (*.ipynb). Both follow the standards below.

---

## Python Style & Naming

### File Naming
- **Python modules**: snake_case, descriptive purpose
  - `salt3_common.py` ✓ (centralized utilities)
  - `utils.py` ✗ (too generic)
- **Jupyter notebooks**: numeric prefix + snake_case + purpose
  - `00_colab_runbook.ipynb` ✓
  - `01_init_embeddings_salt3.ipynb` ✓
  - `notebook_1.ipynb` ✗

### Variable & Function Naming
- **Functions/variables**: snake_case
  ```python
  def load_model_safe(name_or_path: str | Path, ...) -> PreTrainedModel:
      pass
  
  def extract_embedding_weight(model: nn.Module) -> torch.Tensor:
      pass
  ```
- **Classes**: PascalCase
  ```python
  class NeoMLMTrainer(Trainer):
      pass
  
  class NeoBERTSequenceClassifier(nn.Module):
      pass
  ```
- **Constants**: UPPER_SNAKE_CASE
  ```python
  DEFAULT_VOCAB_SIZE = 32064
  FASTTEXT_URL = "https://..."
  ```

### Docstring Format
Use Google-style docstrings with clear type hints:

```python
def make_mlm_datasets(
    tokenizer,
    cache_dir: str | Path,
    num_examples: int,
    max_seq_len: int,
    num_chunks: int | None = None,
    eval_ratio: float = 0.02,
    seed: int = 42,
):
    """Tokenize CulturaX-vi, cache to Arrow, return train/eval split.

    The cache key is (num_examples, max_seq_len, tokenizer_fingerprint) so
    streaming + tokenization only happens once per unique config. After
    caching, num_chunks controls how many chunks are actually used.

    Args:
        tokenizer: HF tokenizer (must be pruned SALT tokenizer).
        cache_dir: Root directory for dataset caches.
        num_examples: Number of CulturaX documents to stream.
        max_seq_len: Chunk length in tokens.
        num_chunks: Optional cap on chunks to use. None = all.
        eval_ratio: Fraction for validation split.
        seed: Random seed for train/eval split.

    Returns:
        (train_dataset, eval_dataset) as HF Dataset objects.

    Raises:
        RuntimeError: If streaming fails (network error, deprecated dataset).
        ValueError: If cache is corrupted or vocab mismatch detected.
    """
```

### Type Hints
- Use Python 3.10+ union syntax: `str | Path` not `Union[str, Path]`
- Annotate all function parameters and returns
- Annotate class attributes in `__init__`

```python
class NeoBERTSequenceClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base_model: nn.Module = base_model
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.classifier: nn.Linear = nn.Linear(hidden_size, num_labels)
```

---

## Module Organization

### salt3_common.py Structure

Logical groupings (top-to-bottom):

| Section | Lines | Purpose |
|---|---|---|
| **Google Colab** | 18–40 | Drive, path resolution |
| **File I/O** | 42–85 | JSON, JSONL, path utilities |
| **Seeding & Env** | 18–54 | RNG, HF config |
| **Model Loading** | 195–249 | Safe loading, graceful error handling |
| **Embedding Utils** | 251–312 | Weight extraction, fingerprinting |
| **Dataset Utils** | 315–328 | Tokenization sanity |
| **Trainer Classes** | 380–648 | Custom Trainer, collators, args |
| **Model Wrappers** | 409–457 | NeoBERTSequenceClassifier, QA |
| **Diagnostics** | 662–770 | Health checks, embedding stats |

Each section is separated by blank lines and optional comments:

```python
# ── Model Loading ──────────────────────────────────────────────────────
def load_model_safe(...):
    pass


# ── Embedding Utils ────────────────────────────────────────────────────
def extract_embedding_weight(...):
    pass
```

### Notebook Structure

Each notebook follows a standard template:

1. **Markdown cell**: Title + high-level purpose
2. **Pip cell**: Dependency install
3. **Imports & setup cell**: Module imports, environment config, device init
4. **Config cell(s)**: User-configurable parameters (INIT_NAME, RUN_NAME, hyperparams)
5. **Functional cells**: One logical task per cell (e.g., "Load models", "Mine anchors", "SALT projection")
6. **Output cells**: Save artifacts, verify integrity
7. **Verification cells**: Spot checks, sanity tests

---

## Import Conventions

### Organization
```python
# Standard library
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, Trainer
from datasets import Dataset, load_dataset

# Try-except for optional libraries
try:
    from google.colab import drive
except ImportError:
    drive = None  # Run locally
```

### Avoid Star Imports
```python
# ✓ Good
from transformers import AutoTokenizer, AutoModel

# ✗ Avoid
from transformers import *
```

---

## Error Handling & Logging

### Try-Except Pattern
```python
def load_model_safe(name_or_path: str | Path, ...):
    try:
        # Primary path
        safe_path = hf_hub_download(repo_id=name_or_path, filename="model.safetensors")
        state_dict = load_file(safe_path)
    except Exception:
        # Fallback
        bin_path = hf_hub_download(repo_id=name_or_path, filename="pytorch_model.bin")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  ⚠ {len(result.missing_keys)} MISSING keys (left at random init):")
        for k in result.missing_keys[:20]:
            print(f"      {k}")
```

### Print vs Logging
- **Print for notebooks**: Use formatted strings with visual markers (✓, ✗, ⚠, ──)
- **Print for modules**: Function entry/exit logs, progress bars
- **No logging module**: Keep it simple for research code

```python
print(f"── Loading NeoBERT ──")
print(f"  Model: {model_id}")
print(f"  Device: {device}")
print(f"✓ Model loaded successfully")
```

---

## Design Patterns

### 1. Path Handling
Always use `Path` from `pathlib`, not string paths:

```python
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Usage
model_dir = ensure_dir(project_root / "init" / init_name / "model")
```

### 2. Configuration as Dict
Store all config in dictionaries, save/load as JSON:

```python
run_config = {
    "mode": "new",
    "run_name": run_name,
    "base_model_ref": base_model_ref,
    "learning_rate": 1e-4,
    "batch_size": 32,
}
write_json(run_dir / "run_config.json", run_config)
```

### 3. Fingerprinting for Cache Keys
Hash vocabularies and configs to prevent stale data:

```python
tok_fp = tokenizer_fingerprint(tokenizer)
cache_dir = Path(cache_dir) / f"culturax_vi_{num_examples}_seq{max_seq_len}_tok{tok_fp['short']}"
```

### 4. Graceful Degradation
Detect and report issues without crashing:

```python
if result.missing_keys:
    print(f"  ⚠ load_model_safe: {len(result.missing_keys)} MISSING keys")
    for k in result.missing_keys[:20]:
        print(f"      {k}")
if not result.missing_keys and not result.unexpected_keys:
    print(f"  ✓ load_model_safe: all {len(state_dict)} keys loaded")
```

---

## Google Drive Folder Contract

All paths resolve via `project_root()` utility, which returns:
- `/content/drive/MyDrive/SALT3` if in Colab
- `./SALT3` if local

### Expected Structure
```
{PROJECT_ROOT}/
├── code/
│   ├── salt3_common.py
│   └── *.ipynb
├── init/
│   └── {INIT_NAME}/
│       └── model/         # HF model artifact (config.json, *.safetensors, tokenizer.json, etc.)
├── runs/
│   └── {RUN_NAME}/
│       ├── final_model/   # Trained model (save_pretrained output)
│       ├── checkpoints/   # Intermediate checkpoints (trainer artifacts)
│       ├── metrics.jsonl  # Streamed metrics (one JSON per line)
│       ├── plots/         # PNG plots (matplotlib)
│       └── run_config.json  # Run metadata
└── datasets/
    └── culturax_vi_{num_examples}_seq{max_seq_len}_tok{fingerprint}/
        ├── train/         # Arrow parquets
        └── validation/
```

### Validation
```python
def validate_model_artifact(path: str | Path) -> None:
    """Raise FileNotFoundError if artifact is incomplete."""
    path = Path(path)
    required_any = [path / "model.safetensors", path / "pytorch_model.bin"]
    assert any(p.exists() for p in required_any), "No model weights"
    assert (path / "config.json").exists(), "Missing config.json"
    # Check tokenizer files
```

---

## Notebook-Specific Conventions

### Cell Markdown Headers
Use `##` for major sections, `###` for subsections:

```markdown
## Load Source and Target Models

```python
source_model = load_model_safe(SOURCE_MODEL_ID, model_cls=AutoModelForMaskedLM)
```

### Configuration Cells
Mark with comment header, keep all params in one cell:

```python
# ── Configuration ──────────────────────────────────────────────────────
INIT_NAME = 'videberta_salt_init_v1'
SOURCE_MODEL_ID = 'chandar-lab/NeoBERT'
TARGET_MODEL_ID = 'Fsoft-AIC/videberta-base'
TARGET_VOCAB_SIZE = None  # None = match source
```

### Output Cells
Always end with status summary:

```python
print(f'\n── Summary ──')
print(f'  Init artifact: {INIT_DIR}')
print(f'  Anchor pairs: {len(anchor_map)}')
print(f'  Embedding shape: {new_embeddings.shape}')
print(f'  Next: Load this model in 02_train_cpt_run.ipynb with BASE_MODEL_REF = "init/{INIT_NAME}/model"')
```

---

## Testing & Validation Checklist

### Pre-Commit (salt3_common.py)
- [ ] All functions have type hints and docstrings
- [ ] No hardcoded paths (use `Path`, `project_root()`)
- [ ] Imports organized: stdlib, third-party, optional
- [ ] Error messages are descriptive (include context, suggestions)

### Pre-Run (Notebooks)
- [ ] Config cell is at the top, all params clearly named
- [ ] Google Drive mounted (if Colab)
- [ ] Folders created via `ensure_dir()`, not raw `mkdir`
- [ ] Model loading uses `load_model_safe()`
- [ ] Dataset cache keys include tokenizer fingerprint

### Post-Run (All Outputs)
- [ ] Artifacts saved with validation: `validate_model_artifact()`
- [ ] Metrics logged to JSONL (not terminal only)
- [ ] Embedding round-trip error < 1e-5 (safetensors)
- [ ] Forward pass produces finite logits (no NaN)
- [ ] Special tokens match source model exactly
- [ ] Anchor spot-checks pass (10 random pairs)

---

## Performance Conventions

### Memory Efficiency
- Move large tensors to device *only when needed*
- Detach and send to CPU before serializing

```python
# ✓ Good
source_embeddings = source_embeddings.to(DEVICE)
new_embeddings[vi_new_id] = torch.matmul(target_emb, X_emb).squeeze(0)
new_embeddings = new_embeddings.cpu()

# ✗ Avoid
new_embeddings = torch.zeros(..., device=DEVICE).to(DEVICE)  # double .to()
```

### Batch Processing
Use batch size constants, log progress:

```python
BATCH_SIZE = 64
for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i : i + BATCH_SIZE]
    # Process batch
```

### Checkpointing
Resume from last checkpoint in training, don't restart:

```python
checkpoint = get_last_checkpoint(trainer.args.output_dir) if mode == "resume" else None
trainer.train(resume_from_checkpoint=checkpoint)
```

---

## Documentation Standards

### Comments in Code
- Explain *why*, not *what* (code shows what it does)
- Keep comments close to the code they describe
- Use `#` for inline, `"""` for docstrings

```python
# ✓ Good: explains the reasoning
# SALT projects non-shared embeddings locally to NeoBERT space.
# We use Sparsemax (not top-k) to ensure sparse support, preventing
# overfitting to a fixed neighbor set when the anchor is far from NeoBERT space.
sparse_support = sparsemax(sim, dim=0)

# ✗ Bad: states the obvious
# Apply sparsemax
sparse_support = sparsemax(sim, dim=0)
```

### Docstring Examples
Include runnable examples for complex functions:

```python
def tokenizer_fingerprint(tokenizer, sample_texts: list[str] | None = None) -> dict[str, Any]:
    """Compute SHA256 hash of tokenizer vocab for cache key.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tok = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")
        >>> fp = tokenizer_fingerprint(tok)
        >>> print(fp["short"])  # First 12 chars of SHA256
        'a1b2c3d4e5f6'
    """
```

---

## Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| Hardcoded `/content/drive` paths | Use `project_root()` |
| `torch.load(..., weights_only=False)` in notebooks | Use `weights_only=True` for security |
| Loading tokenizer without `use_fast` param | Specify explicitly: `use_fast=True` |
| Missing `nan` checks on logits | Always validate: `assert torch.isfinite(logits).all()` |
| Stale dataset cache | Include tokenizer fingerprint in cache key |
| Model loaded but LM head is random | Use `validate_model_artifact()` + `diagnose_model_health()` |
| Training loss diverges | Check MLM collator (`mask_replace_prob=1.0`), learning rate |

---

## Version Control Conventions

### Commit Messages
Use conventional commits format:

```
feat(salt): add anchor mining 3-tier translation strategy
fix(trainer): handle NeoBERT forward signature in NeoMLMTrainer.compute_loss
docs(readme): update setup instructions for Colab
refactor(common): split utilities into modules
```

### Branches
- `main`: Stable, verified code
- `feature/...`: New feature or experiment
- `fix/...`: Bug fix

### What NOT to Commit
- `.ipynb_checkpoints/` (Jupyter temp)
- Google Drive tokens (`.env`, `token.json`)
- Large model files (`.bin`, `.safetensors` > 5GB)
- Notebooks with cell outputs (strip before committing)

---

## Code Review Checklist

- [ ] Type hints on all functions
- [ ] Docstrings with Args, Returns, Raises
- [ ] No bare `except:` (always specify exception type)
- [ ] Logging/print statements have context (what, where, why)
- [ ] Error messages guide next steps ("use load_model_safe()")
- [ ] No magic numbers (use named constants)
- [ ] Function under 50 lines (except data pipelines)
- [ ] Imports are organized and minimal
- [ ] Paths use `Path` not strings
- [ ] Tests pass, no new warnings
