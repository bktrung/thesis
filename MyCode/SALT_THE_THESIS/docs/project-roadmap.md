# Project Roadmap: Vietnamese NeoBERT via SALT

## Project Vision

Demonstrate that **SALT (Semantic Aware Linear Transfer)** can successfully adapt NeoBERT, a state-of-the-art English encoder, to Vietnamese with minimal computational cost. The adapted model should:
1. Initialize embeddings via ViDeBERTa transfer (no training from scratch)
2. Adapt to Vietnamese via continual pre-training on CulturaX-vi
3. Achieve competitive performance on Vietnamese downstream tasks (sentiment, NLI)
4. Cost significantly less than training a 250M-param model from scratch

---

## Phases

### Phase 0: Prototype & Proof-of-Concept (Current)
**Status**: In Progress  
**Duration**: ~2 weeks  
**Owner**: Thesis author

**Objectives**:
- [ ] Fix current library breakage (transformers/xformers/torch version conflicts)
- [ ] Run Stage 1 (SALT init) to completion without errors
- [ ] Run Stage 2 (CPT) on 400K-doc subset, verify no divergence
- [ ] Run Stage 3 (downstream eval) on UIT-VSFC, compare vs baseline

**Deliverables**:
- ✓ Working SALT init pipeline (init artifact saved)
- ✓ CPT training loop converging on 400K docs
- ✓ Downstream eval script with baseline comparison
- ✓ Documentation: README, architecture, code standards

**Known Blockers**:
- Library version coupling (HF transformers + xformers + torch 2.5)
- HuggingFace remote code fragility (`trust_remote_code=True`)
- Translation API rate limits in anchor mining

**Success Criteria**:
- Init artifact builds without errors, embedding fidelity < 1e-5
- CPT loss converges (no NaN, no divergence)
- UIT-VSFC macro F1 > 0.5 (better than random)

---

### Phase 1: Scale & Stabilize (Weeks 3–4)
**Status**: Planned  
**Owner**: Thesis author

**Objectives**:
- [x] Build staged WSD CPT harness (2026-06-17): one reused ceiling cache, windowed sessions, crash/continue resume, hybrid milestones, WSD LR with no re-warm on resume — `16_staged_wsd_cpt.ipynb` + `salt3_staged_*.py` (plan `260617-2021`; 21 CPU smoke checks pass, Colab GPU run pending for empirical Tier-B)
- [ ] Fix library breakage: vendor NeoBERT code locally, pin transformers version
- [ ] Scale CPT to full 3M-doc CulturaX-vi (via the staged harness above)
- [ ] Add comprehensive downstream eval suite (UIT-VSFC, XNLI-vi, semantic similarity)
- [ ] Multi-seed evaluation (3+ seeds) for all tasks
- [ ] Baseline comparison: PhoBERT-base-v2, mBERT, XLM-R

**Deliverables**:
- ✓ Vendored NeoBERT code (no more remote code downloads)
- ✓ Full-scale CPT training (3M docs, ~200K steps)
- ✓ Evaluation results table (SALT vs baselines)
- ✓ Training curves, metrics analysis

**Risks**:
- 3M-doc CPT may take 24+ hours; requires GPU access
- Downstream eval may show SALT underperforming (method debugging needed)
- Cache misses if tokenizer changes mid-phase

**Success Criteria**:
- CPT trains to 200K steps without divergence
- XNLI-vi macro F1 ≥ 0.40 (vs PhoBERT baseline ≈ 0.60)
- Documentation updated with full results

---

### Phase 2: Optimization & Ablation (Weeks 5–6)
**Status**: Planned  
**Owner**: Thesis author + optional collaborators

**Objectives**:
- [ ] Ablation studies: impact of anchor mining tiers, sparsemax vs top-k, norm calibration
- [ ] Hyperparameter tuning: learning rate, warmup, MLM probability
- [ ] Analysis: embedding norm distributions, projection fidelity, training stability
- [ ] Compare CPT with cold-start random init (sanity check)

**Deliverables**:
- ✓ Ablation report (which components matter most?)
- ✓ Tuning report (best hyperparams found)
- ✓ Visualization: embedding norms over training, anchor quality metrics
- ✓ Write-up: lessons learned

**Risks**:
- Ablations may require 10+ separate CPT runs (GPU-intensive)
- Hyperparameter grid search is exponential; must be strategic

**Success Criteria**:
- Identify 1–2 critical SALT components via ablation
- Downstream performance improves by 5–10% via tuning
- Clear recommendations for future practitioners

---

### Phase 3: Modularization & Reproducibility (Week 7+)
**Status**: Planned  
**Owner**: Thesis author

**Objectives**:
- [ ] Refactor salt3_common.py (771 LOC monolith) into modules:
  - `salt_init.py`: Anchor mining, projection, init artifact handling
  - `salt_training.py`: NeoMLMTrainer, dataset pipeline, metrics
  - `salt_eval.py`: Downstream task setup, evaluation utils
  - `salt_diagnostics.py`: Model health checks, fingerprinting
- [ ] Script-based workflow: replace manual Jupyter orchestration
  - `scripts/run_init.py`: Stage 1 (with CLI args)
  - `scripts/run_cpt.py`: Stage 2 (with CLI args)
  - `scripts/run_eval.py`: Stage 3 (with CLI args)
- [ ] Unit tests for critical functions (anchor mining, projection)
- [ ] CI/CD pipeline: automated tests on push

**Deliverables**:
- ✓ Modular codebase (<200 LOC per module)
- ✓ CLI scripts (runnable from shell)
- ✓ Test suite (>80% coverage)
- ✓ Updated documentation

**Risks**:
- Refactoring may introduce bugs; thorough testing needed
- Modularity may reduce notebook interactivity (trade-off)

**Success Criteria**:
- All tests pass, coverage > 80%
- Scripts produce identical results to notebooks
- Code review by 2+ peers (if collaborative)

---

### Phase 4: Publication & Release (Week 8+)
**Status**: Planned  
**Owner**: Thesis author + advisors

**Objectives**:
- [ ] Write thesis paper (3000–5000 words) documenting:
  - SALT method overview
  - Vietnamese NeoBERT adaptation (experiments, results)
  - Ablation studies and analysis
  - Comparison with baselines (PhoBERT, mBERT, XLM-R)
  - Limitations and future work
- [ ] Release code on GitHub (open source)
  - MIT or CC-BY license
  - PyPI package (optional)
- [ ] Release model checkpoints on HuggingFace Hub
  - init artifact (Vietnamese NeoBERT SALT-initialized)
  - Final CPT model (trained on 3M docs)
- [ ] Write blog post or preprint on arXiv

**Deliverables**:
- ✓ Thesis document (PDF)
- ✓ GitHub repo (README, setup instructions, examples)
- ✓ HuggingFace Hub models (2 checkpoints)
- ✓ Blog post or arXiv paper

**Success Criteria**:
- Thesis accepted by advisor(s)
- Code gets 10+ GitHub stars
- Model checkpoint downloaded 100+ times

---

## Timeline

```
Week 1–2   Phase 0: Prototype & Fix        [████████████] CURRENT
Week 3–4   Phase 1: Scale & Stabilize      [░░░░░░░░░░░░] Planned
Week 5–6   Phase 2: Optimization            [░░░░░░░░░░░░] Planned
Week 7     Phase 3: Modularization          [░░░░░░░░░░░░] Planned
Week 8+    Phase 4: Publication             [░░░░░░░░░░░░] Planned
```

---

## Known Issues & Mitigation

| Issue | Severity | Mitigation |
|---|---|---|
| **HF remote code fragility** | High | Vendor model.py, rotary.py locally (Phase 1) |
| **Library version coupling** | High | Pin torch 2.5, transformers 4.46, xformers 0.0.28 (Phase 0) |
| **salt3_common.py monolith** | Medium | Modularize in Phase 3 (< 200 LOC per module) |
| **Translation API rate limits** | Medium | Cache translations, provide fallback anchors (Phase 1) |
| **Google Drive dependency** | Medium | Support local file system via `project_root()` (already done) |
| **Embedding norm calibration** | Low | Monitor during CPT; adjust heuristic if needed (Phase 2) |

---

## Success Metrics

### Quantitative
- **Init artifact fidelity**: Embedding round-trip error < 1e-5
- **CPT convergence**: Loss decreases monotonically, no NaN at 200K steps
- **Downstream performance**:
  - UIT-VSFC: macro F1 ≥ 0.60 (vs PhoBERT ≈ 0.68)
  - XNLI-vi: macro F1 ≥ 0.40 (vs PhoBERT ≈ 0.60)
- **Computational efficiency**: CPT 3M docs in <24 GPU hours (A100 40GB)

### Qualitative
- **Code quality**: Clear naming, comprehensive docstrings, <200 LOC per module
- **Documentation**: README, architecture, code standards, inline comments
- **Reproducibility**: Deterministic results with seed control, version pinning
- **Generalizability**: Method works on other language pairs (e.g., English→Chinese)

---

## Dependencies & Constraints

### Hard Dependencies
- **GPU**: A100 40GB recommended (3M-doc CPT needs ~30GB VRAM)
- **Disk space**: 12GB for cached dataset + 2GB for models
- **Internet**: HuggingFace Hub access, Google Translate API (rate-limited)
- **Time**: ~200 GPU hours for full pipeline (init + CPT + eval)

### Soft Constraints
- **Tokenizer stability**: Changes invalidate dataset cache (mitigated by fingerprinting)
- **Translation quality**: 3-tier mining depends on MarianMT + Google Translate accuracy
- **Anchor availability**: Sparse anchors (< 100) may degrade projection quality

---

## Alternative Approaches Considered

### 1. Monolingual BERT from Scratch
- **Pros**: Clean slate, optimize for Vietnamese
- **Cons**: 250M params × 2.1T tokens ≈ $100K GPU cost; out of scope
- **Decision**: Rejected; SALT is much cheaper

### 2. Fine-Tuning PhoBERT on CulturaX-vi
- **Pros**: PhoBERT already optimized for Vietnamese
- **Cons**: Not a novel contribution; doesn't test SALT method
- **Decision**: Rejected; we want to test SALT on NeoBERT specifically

### 3. XLM-R → Vietnamese (via SALT)
- **Pros**: Multi-lingual, larger model (550M params)
- **Cons**: Slower, harder to reach convergence
- **Decision**: Deferred to Phase 4 (future work)

### 4. Knowledge Distillation Instead of SALT
- **Pros**: Direct teacher→student transfer
- **Cons**: Requires large labeled dataset; SALT is unsupervised
- **Decision**: Rejected; SALT is more practical

---

## Future Work (Beyond Phase 4)

1. **Other language pairs**: English NeoBERT → Chinese, Japanese, Korean
2. **Larger donors**: XLM-R (550M), mBERT (multiple languages)
3. **Multi-lingual SALT**: Transfer from multi-lingual ViDeBERTa
4. **Task-specific SALT**: Anchor mining guided by downstream task (e.g., sentiment)
5. **Continual learning**: Update SALT embeddings during CPT (not frozen)
6. **Efficient SALT**: GPU-accelerated anchor mining (currently CPU-slow)

---

## Resources & Contacts

| Role | Name | Contact |
|---|---|---|
| Thesis Author | [Your Name] | [Email/GitHub] |
| Advisor | [Advisor Name] | [Email] |
| Collaborator (optional) | [If applicable] | [Email/GitHub] |

**Key References**:
- SALT paper: arXiv:2505.10945v2
- NeoBERT paper: arXiv:2502.19587
- ViDeBERTa paper: (see papers/ directory)
- Project repo: (GitHub URL, once released)

---

## Change Log

| Date | Phase | Changes |
|---|---|---|
| 2025-06-07 | 0 | Initial roadmap created; Phase 0 in progress |
| TBD | 1 | Library version fixes, 3M-doc CPT completion |
| TBD | 2 | Ablation studies, hyperparameter tuning |
| TBD | 3 | Modularization, unit tests |
| TBD | 4 | Thesis write-up, GitHub release, HuggingFace Hub |
