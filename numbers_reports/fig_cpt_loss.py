"""Hinh 4.4 (thesis): training curve of the WSD continued-pretraining stage of ViNeoBERT.

Panel (a): training loss vs. tokens seen over the FULL run = stable trunk
           (metrics.jsonl) + final WSD decay/cooldown (metrics_5m_decay.jsonl),
           with reference lines log|V| (~10.3, random) and unigram entropy (~7.3).
Panel (b): eval loss and perplexity vs. tokens seen at the cooled milestones
           (milestone_eval.json, 0.1..3.9B) plus the final cooled checkpoint (~4.8B).
Output: images/cpt_loss_curve.png (300 dpi). English labels, decimal points.
"""
import json
import math
from bisect import bisect_left
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "images" / "cpt_loss_curve.png"

VOCAB = 30522
LOGV = math.log(VOCAB)      # ~10.33 — random-guess loss over the vocabulary
H_UNIGRAM = 7.3            # Vietnamese unigram entropy (Ch.3, sec 3.4.6)
TOK_PER_STEP = 524288      # 512 seqs x 1024 tokens (verified from metrics.jsonl)


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).open() if l.strip()]


# ---- training loss vs tokens: stable trunk + decay ----
trunk = [r for r in load_jsonl(ROOT / "metrics.jsonl")
         if "loss" in r and "tokens_seen" in r]
tok = [r["tokens_seen"] / 1e9 for r in trunk]
loss = [r["loss"] for r in trunk]

decay = [r for r in load_jsonl(ROOT / "metrics_5m_decay.jsonl")
         if r.get("phase") == "decay" and "loss" in r]
dtok = [r["global_step"] * TOK_PER_STEP / 1e9 for r in decay]
dloss = [r["loss"] for r in decay]

# full curve for the light rolling mean
ftok = tok + dtok
floss = loss + dloss
W = 5
sm = [sum(floss[max(0, i - W + 1):i + 1]) / len(floss[max(0, i - W + 1):i + 1])
      for i in range(len(floss))]

# ---- eval milestones: map docs -> tokens seen via metrics.jsonl ----
ds = [r["docs_seen"] for r in trunk]
ts = [r["tokens_seen"] for r in trunk]


def docs_to_btokens(d):
    i = bisect_left(ds, d)
    if i == 0:
        return ts[0] / 1e9
    if i >= len(ds):
        return ts[-1] / 1e9
    f = (d - ds[i - 1]) / (ds[i] - ds[i - 1])
    return (ts[i - 1] + f * (ts[i] - ts[i - 1])) / 1e9


mil = json.loads((ROOT / "milestone_eval.json").read_text())
mtok = [docs_to_btokens(m["milestone_docs"]) for m in mil]
mloss = [m["eval_loss"] for m in mil]
mppl = [m["perplexity"] for m in mil]

# final cooled checkpoint = milestone_5000000_decay, held-out post-eval
# (measured; from the 5M fresh-data WSD-decay run log, chunk 4,674,280 -> 4.786B tokens)
FINAL_TOK = 4674280 * 1024 / 1e9   # ~4.786B tokens
FINAL_EVAL_LOSS = 1.1552
FINAL_EVAL_PPL = 3.175
mtok.append(FINAL_TOK)
mloss.append(FINAL_EVAL_LOSS)
mppl.append(FINAL_EVAL_PPL)

# ---- plot ----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.2, 4.1))

# Panel (a): training loss vs tokens
axa.axhline(LOGV, ls="--", lw=1.1, color="tab:red", alpha=.8)
axa.axhline(H_UNIGRAM, ls=":", lw=1.1, color="tab:green", alpha=.9)
axa.plot(ftok, floss, color="tab:blue", alpha=.18, lw=.8)
axa.plot(ftok, sm, color="tab:blue", lw=1.9)
axa.annotate(f"start $\\approx$ {loss[0]:.1f}", (tok[0], loss[0]),
             xytext=(12, -2), textcoords="offset points", fontsize=9)
axa.annotate(f"end $\\approx$ {sm[-1]:.1f}", (ftok[-1], sm[-1]),
             xytext=(-4, 16), textcoords="offset points", fontsize=9, ha="right")
axa.text(ftok[-1], LOGV, r" $\log|V|\approx 10.3$ (random)",
         va="bottom", ha="right", fontsize=8.5, color="tab:red")
axa.text(ftok[-1], H_UNIGRAM, r" unigram entropy $\approx 7.3$",
         va="bottom", ha="right", fontsize=8.5, color="tab:green")
axa.set_xlabel("Tokens seen (billions)")
axa.set_ylabel("MLM training loss")
axa.set_title("(a) Training loss vs. tokens seen")
axa.set_ylim(0, 11)
axa.set_xlim(-0.15, 5.0)
axa.grid(alpha=.25)

# Panel (b): eval loss + perplexity vs tokens seen
axb.plot(mtok, mloss, "o-", color="tab:blue", lw=1.8)
for x, y in zip(mtok, mloss):
    axb.annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points",
                 fontsize=8, ha="center", color="tab:blue")
axb.set_xlabel("Tokens seen (billions)")
axb.set_ylabel("MLM eval loss", color="tab:blue")
axb.tick_params(axis="y", labelcolor="tab:blue")
axb.set_ylim(0, 3.2)
axb.set_xlim(-0.15, 5.0)
axb.grid(alpha=.25)
axb.set_title("(b) Eval loss and perplexity at milestones")

axr = axb.twinx()
axr.plot(mtok, mppl, "s--", color="tab:orange", lw=1.5, alpha=.9)
for x, y in zip(mtok, mppl):
    axr.annotate(f"{y:.1f}", (x, y), xytext=(0, -12), textcoords="offset points",
                 fontsize=8, ha="center", color="tab:orange")
axr.set_ylabel("Perplexity", color="tab:orange")
axr.tick_params(axis="y", labelcolor="tab:orange")
axr.set_ylim(0, 19)

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
print(f"train: start {loss[0]:.3f} @ {tok[0]:.3f}B -> end {sm[-1]:.3f} @ {ftok[-1]:.3f}B")
print("eval pts (Btok, loss, ppl):",
      [(round(a, 2), round(b, 2), round(c, 2)) for a, b, c in zip(mtok, mloss, mppl)])
