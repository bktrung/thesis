"""Hinh (thesis 4.4.3): vi sao chon SALT + can chinh dong bang.

So sanh 3 chien luoc khoi tao {buoc trong so, SALT hai ma tran, SALT + dong bang}
tai cung moc ~100 trieu token CPT:
(a) mat mat MLM tren tap danh gia;
(b) F1 tren UIT-ViQuAD sau tinh chinh (3 seed, thanh loi = do lech chuan).

Nguon so: salt-tied/metrics.jsonl, salt-pertoken-0.1/cpt_summary.json,
salt-pertoken-freeze/milestone_eval.json, hardtask_summary_bakeoff100k.csv.
Xuat: images/init_strategy_100k.png (300 dpi).
"""
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

vn_num = FuncFormatter(lambda v, _: f"{v:g}")

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "images" / "init_strategy_100k.png"

# ---- mat mat @ ~100M token ----
tied_rows = [json.loads(l) for l in (ROOT / "salt-tied" / "metrics.jsonl").open()]
loss_tied = [r["eval_loss"] for r in tied_rows if "eval_loss" in r][-1]
loss_salt = json.load((ROOT / "salt-pertoken-0.1" / "cpt_summary.json").open())["post"]["eval_loss"]
ms = json.load((ROOT / "salt-pertoken-freeze" / "milestone_eval.json").open())
loss_freeze = next(m for m in ms if m["milestone_docs"] == 100000)["eval_loss"]

# ---- MRC F1 @ 100M token ----
mrc = {}
with (ROOT / "hardtask_summary_bakeoff100k.csv").open() as f:
    for r in csv.DictReader(f):
        if r["task"] == "MRC-ViQuAD" and r["metric"] == "f1":
            mrc[r["arm"]] = (float(r["mean"]), float(r["std"]))

# SALT + freeze: F1 tai milestone 100k (checkpoint lam nguoi) cua run chinh —
# nguon images/mrc_summary.md (SALT-100k: 72.15±0.12), KHONG dung so bakeoff cu.
MRC_FREEZE_100K = (72.15, 0.12)

ARMS = [
    ("Weight\ntying", loss_tied, mrc["trung_salt_dectied"], "#999999"),
    ("SALT\n(two matrices)", loss_salt, mrc["trung_salt_decpertoken"], "#6baed6"),
    ("SALT +\nfrozen align", loss_freeze, MRC_FREEZE_100K, "tab:blue"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9))

# ---- (a) mat mat ----
for i, (label, l, _, color) in enumerate(ARMS):
    ax1.bar(i, l, width=0.55, color=color)
    ax1.annotate(f"{l:.2f}"
                 + f"\n(PPL {math.exp(l):.1f})",
                 (i, l), textcoords="offset points", xytext=(0, 4),
                 fontsize=9, ha="center")
ax1.set_xticks(range(len(ARMS)), [a[0] for a in ARMS])
ax1.set_ylim(0, 7.6)
ax1.set_ylabel("Held-out MLM loss")
ax1.set_xlabel("(a)")
ax1.yaxis.set_major_formatter(vn_num)
ax1.grid(axis="y", alpha=0.25)
ax1.spines[["top", "right"]].set_visible(False)

# ---- (b) MRC F1 ----
for i, (label, _, (m, s), color) in enumerate(ARMS):
    ax2.errorbar(i, m, yerr=s, fmt="o", ms=7, color=color, capsize=4, lw=1.6)
    ax2.annotate(f"{m:.2f}", (i, m),
                 textcoords="offset points", xytext=(10, 2), fontsize=9, color=color)
ax2.set_xticks(range(len(ARMS)), [a[0] for a in ARMS])
ax2.set_xlim(-0.5, 2.6)
ax2.set_ylim(48, 76)
ax2.set_ylabel("F1 on UIT-ViQuAD")
ax2.set_xlabel("(b)")
ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.yaxis.set_major_formatter(vn_num)
ax2.grid(axis="y", alpha=0.25)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("saved", OUT)
