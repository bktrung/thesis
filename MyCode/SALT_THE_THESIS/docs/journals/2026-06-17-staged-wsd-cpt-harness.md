# Journal — Staged WSD CPT harness (2026-06-17)

Plan: `plans/260617-2021-staged-wsd-cpt-resumable-milestones/` (5 phases, completed).
Review: `plans/reports/code-review-260617-2110-staged-wsd-cpt-harness-report.md`.

## What shipped

Single-arm, config-driven continued-pretraining harness. One tokenized "ceiling" cache reused across
inits; trains arbitrary per-session doc budgets; resumes after crashes; continues to NEW data windows
(never re-reads doc 0); one budget-sliceable metrics file; milestone snapshots; WSD LR that never
re-warms on resume.

Files: `salt3_staged_schedule.py` (WSD LR + cooldown), `salt3_staged_cpt.py` (run_session + atomic save)
+ `_manifest.py` + `_callbacks.py` + `_plot.py`, `16_staged_wsd_cpt.ipynb`, `salt3_staged_cpt_checks.py`.
`salt3_common.py` extended ADDITIVELY (window-slice kwargs on `make_mlm_datasets`, `cache_meta.json`,
`read_cache_meta`, `assert_shared_tokenizer`, docs↔chunks converters). Each module < 200 LOC.

## Decisions

- **Callback-driven WSD LR, not external LambdaLR.** Dev box has no transformers (Colab-only), so HF's
  stepping of an external scheduler is unverifiable. A `TrainerCallback` overwriting
  `optimizer.param_groups['lr']` in `on_step_begin` (with `optimizers=(opt, None)` +
  `lr_scheduler_type="constant"` as inert placeholder) makes the applied LR exact on any version.
- **No re-warm = key on GLOBAL step.** `wsd_lr_at(offset + local_step)`; resume passes
  `offset = manifest.global_step`, so `offset ≥ warmup` ⇒ flat at peak from local step 0. Warmup pinned
  in the manifest at arm creation so it never drifts if `CEILING_DOCS` changes later.
- **Two deliberate deviations from the plan (correctness):** (1) reserved cooldown pool = TAIL of the
  train split — the plan's `POOL_START ≥ CEILING_TRAIN_CHUNKS` yields an EMPTY pool from a ceiling-sized
  cache. (2) Each cooldown forks THAT milestone's stable snapshot (length scaled to the milestone
  budget), not the end-of-session `current_model` — matters when one session spans several table
  milestones. Code review independently confirmed both sound.

## Lesson — the bug the tests couldn't see

Refactoring the cache-key build into `_dataset_cache_dir` left a dangling `cache_key` reference on the
cache-MISS build path (`salt3_common.py:978`). All 18 CPU smoke checks passed and `py_compile` was clean
— because the checks AST-exercise only the cache-HIT *slice* path; the build branch needs `datasets`,
absent locally. Code review caught it (would have crashed the nb16 ceiling build — the foundation of the
whole harness). Fix: `cache_dir.name` (== the old key string). Then added a torch-free **no-undefined-
names AST guard** to the smoke suite so this class of refactor regression can't pass again.

Takeaway: a green test suite that only covers one branch is a false signal — when an env gap forces
branch-selective testing, add a static guard for the branch you can't execute.

## State / next

Verified torch-free: 21/21 CPU smoke checks (Tier A); legacy `make_mlm_datasets` byte-identical
(nb02/nb15 unaffected); all imports resolve. **Pending (env-gated):** a Colab GPU run to confirm Tier-B
(live `WSDLRCallback` no-rewarm, optimizer-state carry) and end-to-end continue/resume/milestone-reload.
