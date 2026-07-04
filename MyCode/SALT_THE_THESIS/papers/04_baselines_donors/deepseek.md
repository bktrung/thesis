---
title: "DeepSeek-V3 Technical Report"
authors: DeepSeek-AI
year: 2024
venue: Technical Report
arxiv: 2412.19437
url: https://arxiv.org/abs/2412.19437
tags: [llm, scaling, moe, modern-architecture, context]
---

# DeepSeek-V3 — a modern large-model reference point

## Core idea
A large open **Mixture-of-Experts (MoE)** language model (671B total parameters, ~37B activated per
token) trained efficiently (~14.8T tokens) with several modern architecture/training advances. Cited
in the thesis as an example of how language models **keep evolving with data scale and architectural
innovation** — the trend context, not a component SALT3 uses.

## Key details (the notable innovations)
- **Multi-head Latent Attention (MLA):** compresses the KV cache via a low-rank latent, cutting
  memory at long context.
- **DeepSeekMoE** with **auxiliary-loss-free load balancing:** balances expert routing via learned
  per-expert bias terms instead of an auxiliary balancing loss, avoiding its quality cost.
- **Multi-Token Prediction (MTP)** training objective for denser signal and speculative decoding.
- **FP8 mixed-precision training** at scale — a large, stable low-precision training run.
- Uses **RoPE** (YaRN-style extension) and the AdamW recipe shared across modern LLMs ([[rope]],
  [[adamw]]).

## Results / why it matters
Reaches frontier-class open-model quality at a fraction of typical training compute, demonstrating
that architecture + systems co-design (MoE + MLA + FP8) — not just scale — drives progress. A canonical
"current state of large LMs" citation.

## How NeoBERT / SALT3 uses this
Purely **contextual / motivational** in the thesis: it appears in the introduction as evidence that
language modeling is advancing rapidly with scale and architecture, motivating a **modern encoder
(NeoBERT)** and its adaptation to Vietnamese. SALT3 does **not** use MoE, MLA, FP8, or MTP — NeoBERT is
a dense bidirectional encoder ([[neobert]]). This note exists so the introductory citation has a
grounded source; it is not a baseline or donor.

## Relation: [[neobert]] [[llama]] [[rope]] [[adamw]] [[modernbert]]
