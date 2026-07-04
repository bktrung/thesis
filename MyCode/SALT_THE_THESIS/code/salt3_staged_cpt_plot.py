"""Budget-sliceable training-curve plotting for staged CPT, separated from the runner so the
orchestration module stays focused (and matplotlib import stays out of the training path)."""
from __future__ import annotations

import math
from pathlib import Path


def plot_by_budget(metrics_path, *, x: str = "docs_seen", max_docs: float | None = None,
                   save_path=None):
    """Plot train/eval loss + eval perplexity against a budget axis from the continuous
    metrics file.

    ``x`` is any budget field logged by the metrics callback (``docs_seen``/``tokens_seen``/
    ``chunks_seen``/``step``). ``max_docs`` slices to an earlier budget so a run that broke or
    over-trained still plots cleanly at any budget you reached. Latest-per-x dedup keeps the
    live trajectory when a resumed session re-logged steps an abandoned one already wrote.
    """
    import matplotlib.pyplot as plt

    from salt3_common import read_jsonl

    rows = read_jsonl(metrics_path)
    if max_docs is not None:
        rows = [r for r in rows if r.get("docs_seen", 0) <= max_docs]

    def latest(pairs):
        by_x: dict[float, float] = {}
        for xv, yv in pairs:
            by_x[xv] = yv
        return sorted(by_x.items())

    train = latest((r[x], r["loss"]) for r in rows if "loss" in r and x in r)
    evals = latest((r[x], r["eval_loss"]) for r in rows if "eval_loss" in r and x in r)
    ppl = [(xv, math.exp(min(yv, 20))) for xv, yv in evals]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, data, title, marker in (
        (axes[0], train, "Train loss", None),
        (axes[1], evals, "Eval loss", "o"),
        (axes[2], ppl, "Eval perplexity", "o"),
    ):
        if data:
            ax.plot([a for a, _ in data], [b for _, b in data], marker=marker)
        ax.set_title(f"{title} vs {x}")
        ax.set_xlabel(x)
        ax.grid(alpha=0.3)
    axes[2].set_yscale("log")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
