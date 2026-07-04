"""Hinh 4.1 (thesis): eval-loss cua 3 phuong an he so co gamma (0.1 / 0.5 / 1.0).

Doc metrics.jsonl cua tung arm, ve duong eval_loss theo buoc huan luyen,
kem 2 duong tham chieu: entropy tu don tieng Viet (~7,3) va log|V| (~10,3).
Xuat: images/gamma_scale_loss.png (300 dpi).
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "images" / "gamma_scale_loss.png"

ARMS = [
    ("salt-pertoken-0.1", r"$\gamma = 0{,}1$ (đề xuất)", "tab:blue", (8, -3)),
    ("salt-pertoken-0.5", r"$\gamma = 0{,}5$", "tab:orange", (8, -12)),
    ("salt-pertoken-1.0", r"$\gamma = 1{,}0$ (không co)", "tab:red", (8, 3)),
]

VOCAB = 30522
LOGV = math.log(VOCAB)          # ~10,33 — muc mat mat cua du doan ngau nhien deu
H_UNIGRAM = 7.3                 # entropy tu don tieng Viet (Ch.3, muc 3.4.6)


def eval_series(arm_dir: Path):
    steps, losses = [], []
    for line in (arm_dir / "metrics.jsonl").open():
        r = json.loads(line)
        if "eval_loss" in r:
            steps.append(r["step"])
            losses.append(r["eval_loss"])
    return steps, losses


fig, ax = plt.subplots(figsize=(7.5, 4.2))

for folder, label, color, offset in ARMS:
    s, l = eval_series(ROOT / folder)
    ax.plot(s, l, marker="o", ms=3.5, lw=1.8, color=color, label=label)
    ax.annotate(f"{l[-1]:.2f}".replace(".", ","), (s[-1], l[-1]),
                textcoords="offset points", xytext=offset,
                fontsize=9, color=color)

ax.axhline(LOGV, ls="--", lw=1, color="gray")
ax.annotate(r"$\log|\mathcal{V}|\approx 10{,}3$ (đoán ngẫu nhiên)",
            (182, LOGV), textcoords="offset points", xytext=(-4, 5),
            fontsize=9, color="gray", ha="right")
ax.axhline(H_UNIGRAM, ls=":", lw=1.2, color="gray")
ax.annotate(r"entropy từ đơn tiếng Việt $\approx 7{,}3$",
            (182, H_UNIGRAM), textcoords="offset points", xytext=(-4, 5),
            fontsize=9, color="gray", ha="right")

ax.set_xlabel("Bước huấn luyện")
ax.set_ylabel("Mất mát MLM trên tập đánh giá")
ax.set_xlim(-3, 215)
ax.set_ylim(2.5, 11.2)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left")

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("saved", OUT)
