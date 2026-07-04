---
title: "YaRN: Efficient Context Window Extension of Large Language Models"
authors: Peng, Quesnelle, Fan, Shippole
year: 2023
venue: ICLR 2024
arxiv: 2309.00071
url: https://arxiv.org/abs/2309.00071
tags: [rope, context-extension, positional, long-context]
---

# YaRN — extending RoPE context windows cheaply

## Core idea
A method to **extend the usable context length** of a model that was pre-trained with **Rotary
Position Embeddings (RoPE)** ([[rope]]) far beyond its training length, with **very little
fine-tuning** (orders of magnitude fewer tokens than training from scratch at the long length). YaRN
= "Yet another RoPE extensioN".

## Key details
- Naive **Position Interpolation (PI)** linearly rescales all RoPE frequencies — but it degrades
  high-frequency (local) information.
- YaRN uses **NTK-by-parts interpolation**: interpolate **low-frequency** dimensions (which need to
  span the longer range) while leaving **high-frequency** dimensions (local detail) nearly
  untouched, ramping between them.
- Adds a **temperature / attention-scaling** term on the attention logits to compensate for the
  changed distribution at long range.
- Requires only a short fine-tune on long sequences to adapt.

## Results / why it matters
Reaches long context (e.g. tens of thousands of tokens) with a small fraction of the compute of
retraining, while preserving short-context quality better than plain PI. Became a standard recipe for
turning a RoPE model into a long-context model.

## How NeoBERT / SALT3 uses this
NeoBERT uses **RoPE** ([[rope]]) and natively targets a **4,096-token** context. YaRN appears in the
thesis as evidence that **RoPE positions can be recalibrated/interpolated to support longer contexts**
— i.e. the positional scheme NeoBERT (and therefore ViNeoBERT) inherits is extensible. Per
[[salt3-method-verified]], any 4,096-context extension work is deferred/user-handled; YaRN is cited as
the background technique that makes such extension feasible, not something SALT3 itself runs.

## Relation: [[rope]] [[neobert]] [[transformer]]
