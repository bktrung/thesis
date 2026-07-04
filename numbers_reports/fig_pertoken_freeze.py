"""Hinh 4.2 (thesis): vi sao chon SALT theo token + can chinh dong bang.

(a) F1 tren UIT-ViQuAD sau ~100 trieu token CPT (3 seed, thanh loi = do lech chuan)
    cho 3 phuong an SALT + duong tham chieu doi chung ngau nhien tot nhat.
(b) Mat mat MLM tren tap danh gia tai cung moc ngan sach: theo token vs + dong bang.

Nguon so: hardtask_summary_bakeoff100k.csv, salt-pertoken-0.1/cpt_summary.json,
salt-pertoken-freeze/milestone_eval.json.
Xuat: images/pertoken_freeze_100k.png (300 dpi).
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

vn_num = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "images" / "pertoken_freeze_100k.png"

# ---- doc so lieu ----
mrc = {}
with (ROOT / "hardtask_summary_bakeoff100k.csv").open() as f:
    for r in csv.DictReader(f):
        if r["task"] == "MRC-ViQuAD" and r["metric"] == "f1":
            mrc[r["arm"]] = (float(r["mean"]), float(r["std"]))

loss_pertoken = json.load((ROOT / "salt-pertoken-0.1" / "cpt_summary.json").open())
ms = json.load((ROOT / "salt-pertoken-freeze" / "milestone_eval.json").open())
loss_freeze = next(m for m in ms if m["milestone_docs"] == 100000)

ARMS = [
    ("trung_salt_globalmap_freqbias", "Ánh xạ\ntoàn cục", "tab:orange"),
    ("trung_salt_decpertoken", "Theo token", "#6baed6"),
    ("trung_salt_freezealigned", "Theo token\n+ đóng băng", "tab:blue"),
]
RANDOM_BEST = mrc["trung_random_meannorm"][0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9), width_ratios=[1.5, 1])

# ---- (a) MRC F1 ----
for i, (arm, label, color) in enumerate(ARMS):
    m, s = mrc[arm]
    ax1.errorbar(i, m, yerr=s, fmt="o", ms=7, color=color, capsize=4, lw=1.6)
    ax1.annotate(f"{m:.2f}".replace(".", ","), (i, m),
                 textcoords="offset points", xytext=(10, 2), fontsize=9, color=color)
ax1.axhline(RANDOM_BEST, ls="--", lw=1, color="gray")
ax1.annotate("đối chứng ngẫu nhiên tốt nhất"
             f" ({RANDOM_BEST:.1f})".replace(".", ","),
             (1.0, RANDOM_BEST), textcoords="offset points", xytext=(0, 5),
             fontsize=9, color="gray", ha="center")
ax1.set_xticks(range(len(ARMS)), [a[1] for a in ARMS])
ax1.set_xlim(-0.5, 2.6)
ax1.set_ylim(52, 74)
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_major_formatter(vn_num)
ax1.set_ylabel("F1 trên UIT-ViQuAD")
ax1.set_xlabel("(a)")
ax1.grid(axis="y", alpha=0.25)
ax1.spines[["top", "right"]].set_visible(False)

# ---- (b) eval loss tai moc ~100 trieu token ----
pairs = [
    ("Theo token", loss_pertoken["post"]["eval_loss"],
     loss_pertoken["post"]["eval_perplexity"], "#6baed6"),
    ("Theo token\n+ đóng băng", loss_freeze["eval_loss"],
     loss_freeze["perplexity"], "tab:blue"),
]
for i, (label, l, ppl, color) in enumerate(pairs):
    ax2.bar(i, l, width=0.55, color=color)
    ax2.annotate(f"{l:.2f}".replace(".", ",") + f"\n(PPL {ppl:.1f})".replace(".", ","),
                 (i, l), textcoords="offset points", xytext=(0, 4),
                 fontsize=9, ha="center", color="black")
ax2.set_xticks(range(2), [p[0] for p in pairs])
ax2.set_xlim(-0.6, 1.6)
ax2.set_ylim(0, 3.9)
ax2.yaxis.set_major_formatter(vn_num)
ax2.set_ylabel("Mất mát MLM trên tập đánh giá")
ax2.set_xlabel("(b)")
ax2.grid(axis="y", alpha=0.25)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("saved", OUT)
