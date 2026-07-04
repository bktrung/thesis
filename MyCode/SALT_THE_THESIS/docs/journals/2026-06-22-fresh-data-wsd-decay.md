# Fresh-data WSD decay — breaking the MRC plateau

Date: 2026-06-22
Plan: `plans/260622-2015-fresh-culturax-decay-break-mrc-plateau/`
Design: `plans/reports/brainstorm-260622-2006-best-decay-recipe-to-break-mrc-plateau-report.md`

## Diagnosis

MRC (UIT-ViQuAD) F1 plateaus ~75.5–75.8 across 1M–4M CPT docs while MLM eval-loss keeps falling.
Root cause is the **schedule, not a data ceiling**: the staged trunk runs at constant peak LR (1e-4)
with no global decay, and each milestone's reported number comes from a short cosine cooldown over a
tiny **recycled** 20k-chunk pool looped ~15–20×. The phase where downstream quality consolidates (the
LR decay) never sees fresh tokens, so accumulated MLM gains don't cash into F1. The "4M < 3M" dip is
within single-seed noise, not a real regression.

## Recipe (from primary sources)

- **Shape: 1-sqrt → 0** — beats linear > cosine for the cooldown curve (Hägele et al. 2024, 2405.18392).
- **Length ~20%** — compute-optimal cooldown fraction (same paper); validated our earlier guess.
- **Carry optimizer state** — resetting Adam moments hurts; restore from the pre-cooldown checkpoint
  (cooldown-dynamics, 2508.01483). Only valid if the trunk is parked at the snapshot's budget; else
  fresh AdamW + short re-warmup.
- **Fresh data in the decay** — MiniCPM (2404.06395): the decay phase preferentially assimilates what
  it sees, so anneal over unseen tokens.

## Correction worth remembering

Initially proposed annealing on `VTSNLP/vietnamese_curated_dataset` as "high quality." Verified that's
wrong for this case: **CulturaX is already heavily cleaned** (mC4+OSCAR, perplexity filter, MinHash
near-dedup, 46.5% removed) and VTSNLP shares those web sources — the only real differentiator is its
Wikipedia/news register, which is also the UIT-ViQuAD contamination risk (ViQuAD is built from VN
Wikipedia). So **fresh CulturaX is the primary**; HQ-anneal demoted to optional follow-up. The
schedule fix (1-sqrt + carried-opt + fresh) is the high-confidence lever; the data-quality swap is not.

## Implementation

- `salt3_fresh_decay.py` (new): `plan_decay_window` (pure, CPU-tested) + `run_fresh_decay` — forks
  `milestone_4000000_stable`, loads a fresh contiguous unseen window, asserts slice-integrity +
  no-repeat, writes `milestone_5000000_decay/`. **Manifest-neutral** so nb16's flat trunk continuation
  is untouched.
- `salt3_staged_schedule.py`: `onesqrt_lr_at` + shape registry; `make_cooldown_lr_callback(shape,
  warmup_steps)` and `run_cooldown_branch(shape, opt_state_path, warmup_steps)` — cosine/no-carry
  defaults keep all existing callers byte-identical.
- MRC eval: discovery priority `decay > cooled > stable` (`salt3_mrc_models.py`); report stars the
  decay point (`salt3_mrc_report.py`); the benchmark already looped seeds + tagged kind + used a fixed
  split, so 3-seed (42, 123, 456) mean±std needed only `cfg.seeds` + nb17 wiring.
- `19_fresh_data_decay_5m.ipynb` (new): probe gate → optional 6M rebuild+prefix-hash → decay → stitched
  `metrics_5m_decay.jsonl`. Hardware split: decay = A100, MRC eval = L4.

## Status & next

Local code done + reviewed (one dead assertion fixed). Colab runs pending: probe manifest
(`chunks_consumed` decides cache reuse vs rebuild + opt-carry vs fresh), run decay, 3-seed MRC vs a
3-seed 3M baseline. Success = `mean(5M-decay) − mean(3M) > combined std`.

## Open questions

- Plateau may be schedule-hidden gains (likely) or a genuine ceiling (unlikely, ~3% of PhoBERT's
  budget) — this run discriminates.
- A win can't isolate which knob (fresh data vs shape vs optimizer) did it; attribution needs a later
  ablation.
