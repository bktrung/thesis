"""MRC benchmark model registry — 6 pretrained baselines + our SALT milestones.

Spec schema (consumed by salt3_mrc_benchmark.run_benchmark):
  path           : HF hub id OR local milestone dir
  kind           : 'hf' (AutoModelForQuestionAnswering) | 'neobert' (custom QA head)
  family         : grouping label for the summary table
  segment        : True -> VnCoreNLP word-segment inputs (PhoBERT/ViDeBERTa)
  milestone_docs : CPT doc budget for our arms (None for baselines) -> learning curve x-axis

Word-segmentation rationale: PhoBERT/ViDeBERTa were pretrained on word-segmented
text; XLM-R and our syllable-tokenizer NeoBERT were not. Mixing breaks the
comparison, so the flag is pinned per model and recorded in every result row.
"""
from __future__ import annotations

import re
from pathlib import Path

# Working pretrained baselines (HF hub) — the models we benchmark our milestones against.
BASELINE_SPECS: dict[str, dict] = {
    "PhoBERT-base-v2": {"path": "vinai/phobert-base-v2", "kind": "hf", "family": "PhoBERT", "segment": True,  "milestone_docs": None},
    "PhoBERT-large":   {"path": "vinai/phobert-large",   "kind": "hf", "family": "PhoBERT", "segment": True,  "milestone_docs": None},
    "XLM-R-base":      {"path": "FacebookAI/xlm-roberta-base",  "kind": "hf", "family": "XLM-R", "segment": False, "milestone_docs": None},
    "XLM-R-large":     {"path": "FacebookAI/xlm-roberta-large", "kind": "hf", "family": "XLM-R", "segment": False, "milestone_docs": None},
}

# ViDeBERTa is the SALT donor, excluded from the default benchmark because it scores ~2 F1 in
# our harness — cause UNRESOLVED (suspect the segment=True / QA path, NOT the model itself, which
# beats PhoBERT in its paper). Opt-in via build_all_specs(..., include_donor=True) once the
# load/eval path is fixed (diagnose with salt3_videberta_load_probe + nb18).
DONOR_SPECS: dict[str, dict] = {
    "ViDeBERTa-base": {"path": "Fsoft-AIC/videberta-base", "kind": "hf",
                       "family": "ViDeBERTa (donor)", "segment": True, "milestone_docs": None},
}

# Our staged-WSD CPT arm. Milestones are AUTO-DISCOVERED from disk (no hardcoded list to keep in
# sync as training scales to 3M/4M/10M...). For each budget found, the best annealed snapshot wins:
# a fresh-data decay (_decay) is preferred over the recycled-pool cooldown (_cooled), which is
# preferred over the un-cooled trunk (_stable). Mixing kinds on one curve is not apples-to-apples,
# so the chosen ``checkpoint_kind`` is recorded per row.
OUR_ARM = "trung_salt_decpertoken_freezealigned"
_MILESTONE_RE = re.compile(r"milestone_(\d+)_(decay|cooled|stable)$")
_KIND_PRIORITY = ("decay", "cooled", "stable")


def milestone_specs(runs_root: str | Path, arm: str = OUR_ARM) -> dict[str, dict]:
    """Discover every ``milestone_<docs>_(decay|cooled|stable)`` dir with weights under the arm and
    build a neobert spec per budget (decay > cooled > stable; stable falls back with a warning).
    Auto-tracking, so the eval picks up new milestones without edits; a partial Drive sync just
    yields fewer rows."""
    arm_dir = Path(runs_root) / arm
    found: dict[int, dict[str, Path]] = {}
    for p in (sorted(arm_dir.glob("milestone_*")) if arm_dir.exists() else []):
        m = _MILESTONE_RE.match(p.name)
        if m and (p / "model.safetensors").exists():
            found.setdefault(int(m.group(1)), {})[m.group(2)] = p
    specs: dict[str, dict] = {}
    for docs in sorted(found):
        kinds = found[docs]
        kind = next(k for k in _KIND_PRIORITY if k in kinds)   # decay > cooled > stable
        mdir = kinds[kind]
        if kind == "stable":
            print(f"  [warn] {docs:,} docs: no decay/cooled snapshot, using un-cooled trunk ({mdir.name})")
        specs[f"SALT-{docs // 1000}k"] = {
            "path": str(mdir), "kind": "neobert", "family": "SALT (ours)",
            "segment": False, "milestone_docs": docs, "checkpoint_kind": kind}
    if not specs:
        print(f"  [skip] no milestone weights found under {arm_dir}")
    return specs


def build_all_specs(runs_root: str | Path, include_baselines: bool = True,
                    include_milestones: bool = True, include_donor: bool = False) -> dict[str, dict]:
    """Full benchmark registry: working baselines + available milestones. The broken ViDeBERTa
    donor is excluded by default; pass include_donor=True to add it as a reference row."""
    specs: dict[str, dict] = {}
    if include_baselines:
        specs.update(BASELINE_SPECS)
    if include_donor:
        specs.update(DONOR_SPECS)
    if include_milestones:
        specs.update(milestone_specs(runs_root))
    return specs
