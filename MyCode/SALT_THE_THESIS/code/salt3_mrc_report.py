"""MRC benchmark reporting — summary table, markdown table, learning-curve plot.

Consumes the results jsonl written by salt3_mrc_benchmark.run_benchmark
(rows: model, family, segment, milestone_docs, n_params, seed, f1, exact_match).
"""
from __future__ import annotations

from pathlib import Path

from salt3_common import read_jsonl, write_json


def summarize(results_jsonl: str | Path, include_donor: bool = False):
    """-> pandas DataFrame: per-model mean/std of F1 & EM over seeds, sorted by F1. Drops the
    ViDeBERTa donor (family contains 'donor') by default so a stale cached row never pollutes the
    table/plot; pass include_donor=True to keep it as a reference."""
    import pandas as pd

    df = pd.DataFrame(read_jsonl(results_jsonl))
    if df.empty:
        return df
    if not include_donor and "family" in df.columns:
        df = df[~df["family"].fillna("").str.contains("donor", case=False)]
    if "checkpoint_kind" not in df.columns:   # back-compat for rows written before kinds were tagged
        df["checkpoint_kind"] = None
    g = (df.groupby(["model", "family", "milestone_docs"], dropna=False)
           .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                em_mean=("exact_match", "mean"), em_std=("exact_match", "std"),
                n_params=("n_params", "first"), seeds=("seed", "nunique"),
                checkpoint_kind=("checkpoint_kind", "first"))
           .reset_index())
    g["f1_std"] = g["f1_std"].fillna(0.0)
    g["em_std"] = g["em_std"].fillna(0.0)
    return g.sort_values("f1_mean", ascending=False).reset_index(drop=True)


def to_markdown(summary_df, out_path: str | Path | None = None) -> str:
    """Thesis-ready markdown table; optionally written to out_path."""
    lines = ["| Model | Family | Params(M) | EM | F1 |", "|---|---|---|---|---|"]
    for _, r in summary_df.iterrows():
        p = f"{r['n_params'] / 1e6:.0f}" if r.get("n_params") else "-"
        lines.append(f"| {r['model']} | {r['family']} | {p} | "
                     f"{r['em_mean']:.2f}±{r['em_std']:.2f} | {r['f1_mean']:.2f}±{r['f1_std']:.2f} |")
    md = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(md + "\n", encoding="utf-8")
    return md


def plot_learning_curve(summary_df, out_path: str | Path | None = None, metric: str = "f1_mean"):
    """Our SALT milestones (F1 vs CPT docs) with each baseline as a horizontal line."""
    import matplotlib.pyplot as plt

    ours = summary_df[summary_df["milestone_docs"].notna()].sort_values("milestone_docs")
    base = summary_df[summary_df["milestone_docs"].isna()]
    kind = ours["checkpoint_kind"] if "checkpoint_kind" in ours.columns else None

    fig, ax = plt.subplots(figsize=(8, 5))
    if not ours.empty:
        ax.plot(ours["milestone_docs"] / 1e6, ours[metric], "o-", lw=2, color="#1f77b4",
                label="SALT (ours)", zorder=3)
        # overplot any fresh-data decay point with a distinct star so a different cooldown method
        # is not silently read as just another point on the recycled-cooldown curve
        if kind is not None:
            dec = ours[kind == "decay"]
            if not dec.empty:
                ax.plot(dec["milestone_docs"] / 1e6, dec[metric], "*", ms=16, color="#d62728",
                        label="SALT fresh-data decay", zorder=4)
        for _, r in ours.iterrows():
            ax.annotate(f"{r[metric]:.1f}", (r["milestone_docs"] / 1e6, r[metric]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    cmap = plt.get_cmap("tab10")
    for i, (_, r) in enumerate(base.iterrows()):
        ax.axhline(r[metric], ls="--", lw=1.2, color=cmap(i % 10), alpha=0.8,
                   label=f"{r['model']} ({r[metric]:.1f})")
    ax.set_xlabel("CPT budget (M docs)")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("MRC (UIT-ViQuAD) — SALT milestones vs pretrained baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def write_report(results_jsonl: str | Path, out_dir: str | Path, include_donor: bool = False):
    """Convenience: summary csv + markdown + plot in out_dir; returns summary df. Excludes the
    broken ViDeBERTa donor by default (pass include_donor=True to keep it as a reference)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = summarize(results_jsonl, include_donor=include_donor)
    if s.empty:
        print("no results yet")
        return s
    s.to_csv(out_dir / "mrc_summary.csv", index=False)
    to_markdown(s, out_dir / "mrc_summary.md")
    plot_learning_curve(s, out_dir / "mrc_learning_curve.png")
    write_json(out_dir / "mrc_summary.json", s.to_dict(orient="records"))
    print(f"wrote summary ({len(s)} models) -> {out_dir}")
    return s
