---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: Dao, Fu, Ermon, Rudra, Ré
year: 2022
venue: NeurIPS 2022
arxiv: 2205.14135
url: https://arxiv.org/abs/2205.14135
tags: [architecture, attention, systems]
---

# FlashAttention

## Core idea
Standard attention is **memory-bandwidth bound**, not compute bound: materializing the `N×N`
attention matrix in GPU HBM dominates cost. FlashAttention computes **exact** attention without ever
writing that matrix to HBM, by tiling Q/K/V into SRAM and using an **online-softmax** running
update. Result: O(N) memory instead of O(N²), and a large wall-clock speedup — which is what makes
long context (4,096+) practical.

## Key math / architecture details
- **Tiling + recomputation:** stream K/V blocks, keep running max `m` and normalizer `ℓ` for a
  numerically-stable online softmax; rescale the partial output as new blocks arrive. Never store
  the full `S = QKᵀ`.
- **IO-awareness:** optimizes for HBM↔SRAM traffic; backward pass recomputes the attention matrix
  on the fly (cheaper than storing it).
- Exact (not an approximation), unlike sparse/low-rank attention.
- **FlashAttention-2** (2307.08691) improves parallelism/work-partitioning for ~2× more speedup.
- Enables **sequence packing / un-padding** (variable-length batches) for efficiency.

## Results / why it matters
2–4× faster attention and much lower memory; standard in modern training stacks and the reason
context windows grew. Pairs naturally with RoPE for long sequences.

## How NeoBERT / SALT3 uses this
NeoBERT's attention path is **xFormers memory-efficient attention** by default (the same IO-aware
family), with **FlashAttention** available for **sequence packing / un-padding** (see
`NeoBERT/README.md`: install `flash_attn` for un-padding). The thesis notebooks
`10_apply_xformers_fix_and_certify` and the forward-NaN bisections (`09/11/12`) wrestle directly
with this attention backend on Colab. SALT3 trains through it unchanged; it matters because a stable
attention backend is a prerequisite for the CPT runs.

## Relation: [[transformer]] [[rope]] [[neobert]]
