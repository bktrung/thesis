---
title: The Depth-to-Width Interplay in Self-Attention
authors: Levine, Wies, Sharir, Bata, Shashua
year: 2020
venue: NeurIPS 2020
arxiv: 2006.12467
url: https://arxiv.org/abs/2006.12467
tags: [architecture, scaling, depth-width]
---

# Depth-to-width interplay (Levine et al.)

## Core idea
For a self-attention network of a *given parameter budget*, there is an **optimal split between depth (number of layers) and width (hidden size)** — and it is **not** "as deep as possible." The paper gives a theory predicting a **width-dependent transition** between a depth-efficient regime (adding layers helps a lot) and a depth-inefficient regime (extra layers stop paying off), plus explicit quantitative guidance for the best depth-to-width allocation at each size.

## Key details
- Self-attention has a capacity that depends jointly on depth `L` and width `d_x`; below a width threshold, stacking more layers is **depth-inefficient** (the network cannot exploit the extra depth).
- Theory predicts, and ablations (depths 6–48) confirm, an optimal `L` for each width / parameter budget — small models are often **too wide and too shallow** for their budget.
- Gives a recipe: for a target parameter count, pick the depth that lands in the depth-efficient regime rather than maximizing width.

## Why it matters
A principled answer to "how many layers vs how wide" — used to justify the deep-and-narrow shape of modern compact models instead of copying BERT's 12×768 / 24×1024.

## How NeoBERT / SALT3 uses this
This is the **theoretical basis for NeoBERT's depth-to-width choice**: NeoBERT keeps hidden size **768** (plug-and-play with BERT-base) but uses **28 layers** instead of 12, explicitly citing this paper to argue BERT-base sits in the width-inefficient regime. The 250M-parameter budget is allocated toward depth. SALT3 inherits this 28×768 body unchanged — the thesis cites Levine et al. in Ch.2 to explain *why* NeoBERT is shaped the way it is (and why its embeddings, which SALT3 re-initializes, feed a deep-narrow stack). → [[neobert]].

## Relation: [[neobert]] [[transformer]] [[bert]]
