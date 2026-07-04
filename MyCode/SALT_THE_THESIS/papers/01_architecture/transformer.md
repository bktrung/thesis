---
title: Attention Is All You Need
authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
year: 2017
venue: NeurIPS 2017
arxiv: 1706.03762
url: https://arxiv.org/abs/1706.03762
tags: [architecture, foundation]
---

# Transformer (Attention Is All You Need)

## Core idea
Replace recurrence and convolution entirely with **self-attention**. A stack of
identical layers, each = (multi-head self-attention) + (position-wise feed-forward),
with residual connections and normalization around each sub-layer. Fully parallel over
sequence positions → far better hardware utilization than RNNs, and direct O(1)-length
paths between any two tokens.

## Key math / architecture details
- **Scaled dot-product attention:**
  `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`.
  The `1/√d_k` scaling keeps logits in a range where softmax gradients don't vanish.
- **Multi-head attention (MHA):** project Q,K,V into `h` subspaces of size `d_k=d_model/h`,
  attend in parallel, concat, project out with `W_O`. Lets the model attend to multiple
  relation types at once.
- **Position-wise FFN:** `FFN(x) = max(0, xW₁+b₁)W₂+b₂` — two linear layers with ReLU,
  applied identically at every position. Inner width typically `4·d_model`.
- **Positional encoding:** the vanilla model has no notion of order, so it *adds* fixed
  sinusoidal position vectors `PE(pos,2i)=sin(pos/10000^{2i/d})` to the token embeddings.
  (NeoBERT replaces this with RoPE — see below.)
- **Sub-layer wiring:** original paper is **post-norm**: `LayerNorm(x + Sublayer(x))`.
  Modern encoders (incl. NeoBERT) move to **pre-norm** for training stability.
- Encoder–decoder model; **BERT/NeoBERT use the encoder stack only.**

## Results / why it matters
Set new SOTA on WMT'14 EN-DE/EN-FR translation at a fraction of prior training cost, and
became the substrate for essentially all modern LLMs and encoders. The MHA + FFN block is
the unit that every paper in this library modifies rather than replaces.

## How NeoBERT / SALT3 uses this
NeoBERT is a **Transformer encoder**. Every component below is a swap on this base:
sinusoidal PE → RoPE ([[rope]]), ReLU FFN → SwiGLU ([[swiglu]]), LayerNorm → Pre-RMSNorm
([[rmsnorm]], [[pre-ln]]). SALT3 inherits this stack unchanged and only re-initializes the
embedding/decoder rows for a Vietnamese vocabulary, then continues pre-training. The
`W_qkv` (no-bias fused QKV) and `W_o` projections described in the repo's
`NeoBERT/docs/architecture.md` are exactly this block.

## Relation: [[bert]] [[rope]] [[swiglu]] [[rmsnorm]] [[neobert]]
