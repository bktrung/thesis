"""Hinh 4.4 (thesis): F1 UIT-ViQuAD theo luong token CPT (truc token, log).

Duong chinh: ViNeoBERT (band = ±std, 3 seed). Doi sanh chinh: PhoBERT (base-v2,
large) — duong dut dam; XLM-R chi mang tinh tham khao — xam nhat.
Nguon so: images/mrc_summary.md (SALT-100k..5000k + 4 baseline, n=3 seed).
Xuat: images/mrc_curve_tokens.png (300 dpi).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

vn_num = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

OUT = Path(__file__).resolve().parent.parent / "images" / "mrc_curve_tokens.png"

# ViNeoBERT theo ngan sach (ty token ~ trieu tai lieu x 1000)
budget = [0.1, 0.5, 1, 2, 3, 4, 5]
f1     = [72.15, 74.79, 75.54, 75.45, 75.80, 75.19, 76.62]
std    = [0.12, 0.66, 1.44, 0.66, 0.39, 0.13, 0.64]

BASELINES = [
    # (ten, F1, mau, kieu net, do dam, offset-y cua nhan)
    ("PhoBERT-large (77,98)",   77.98, "#8c2d04", (0, (6, 2)), 1.8, 5),
    ("PhoBERT-base-v2 (77,42)", 77.42, "tab:red",  (0, (4, 2)), 1.8, -4),
    ("XLM-R-large (77,92)",     77.92, "#bbbbbb", (0, (1, 2)), 1.2, -11),
    ("XLM-R-base (73,32)",      73.32, "#bbbbbb", (0, (1, 2)), 1.2, -4),
]

fig, ax = plt.subplots(figsize=(8.2, 4.6))

for name, v, color, ls, lw, dy in BASELINES:
    ax.hlines(v, 0.085, 5.4, ls=ls, lw=lw, color=color)
    ax.annotate(name, (5.55, v), fontsize=9, color=color, ha="left",
                va="center", textcoords="offset points", xytext=(0, dy))

ax.errorbar(budget, f1, yerr=std, marker="o", ms=5, lw=2, capsize=3,
            color="tab:blue", label="ViNeoBERT", zorder=5)
ax.fill_between(budget, [m - s for m, s in zip(f1, std)],
                [m + s for m, s in zip(f1, std)], color="tab:blue", alpha=0.12)

ax.annotate("72,15 chỉ sau 0,1 tỷ token", (0.1, 72.15),
            textcoords="offset points", xytext=(8, -14), fontsize=9,
            color="tab:blue")
ax.annotate("76,62", (5, 76.62), textcoords="offset points", xytext=(6, 6),
            fontsize=9, color="tab:blue")

ax.set_xscale("log")
ax.set_xticks(budget)
ax.get_xaxis().set_major_formatter(vn_num)
ax.set_xticks([], minor=True)
ax.set_xlim(0.085, 13)
ax.set_ylim(71, 79)
ax.yaxis.set_major_formatter(vn_num)
ax.set_xlabel("Lượng token tiếp tục tiền huấn luyện (tỷ, thang log)")
ax.set_ylabel("F1 trên UIT-ViQuAD")
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("saved", OUT)
