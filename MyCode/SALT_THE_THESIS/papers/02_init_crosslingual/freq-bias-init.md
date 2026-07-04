---
title: "Frequency-bias initialization of the LM head (synthesis note)"
authors: "Synthesis: Zipf 1949; Lin et al. 2017 (Focal Loss bias init); Press & Wolf 2017 (tied embeddings); Grave et al. 2016 (adaptive softmax)"
year: 2017
venue: synthesis (multiple sources)
arxiv: "1708.02002 (Focal Loss); 1608.05859 (tied embeddings); 1609.04309 (adaptive softmax)"
url: https://arxiv.org/abs/1708.02002
tags: [init, decoder, frequency, synthesis]
---

# Frequency-bias init (the SALT3 decoder-bias contribution)

> Synthesis note: there is no single "freq-bias init" paper; SALT3's freq-bias decoder combines
> three well-established ideas. Sources cited inline.

## Core idea
The LM head computes `logits = h·Wᵀ + b`. Before any training, the **bias `b` alone** can encode the
**unigram prior**: set `b_v = log p(v)` (log token frequency). Then at step 0, with a weak/garbled
`h·Wᵀ`, the model already predicts the **marginal** token distribution — the optimal zero-context
guess — instead of a uniform distribution. This removes a large, easy chunk of the initial loss and
lets training spend its early budget on *context*, not on relearning that "the" / common Vietnamese
particles are frequent. SALT3's predecessors (FOCUS/origin SALT) left this bias at **zero**.

## Key math / architecture details
- **Why log-frequency:** the Bayes-optimal context-free predictor is the unigram distribution
  `p(v)`; a softmax whose bias equals `log p(v)` (up to a constant) reproduces it exactly when the
  weight contribution is ~0. Grounded in **Zipf's law** (token frequencies are highly skewed, so a
  good prior is worth a lot) — *Zipf, Human Behavior and the Principle of Least Effort, 1949.*
- **Bias-init precedent:** initializing an output **bias to a log-prior** to stabilize early training
  is exactly the trick in **Focal Loss / RetinaNet** (Lin et al. 2017, arXiv:1708.02002), where the
  final-layer bias is set to `−log((1−π)/π)` so training doesn't diverge from the rare-class
  imbalance. Same principle, classification-over-vocabulary instead of objectness.
- **Counting:** SALT3 estimates `p(v)` by **Laplace-smoothed** unigram counts over the tokenized
  CulturaX-vi corpus: `p(v) = (count_v + 1)/(N + V)`, `b_v = log p(v)` — see
  `vietnamese_unigram_logfreq()` in `scripts/test_decoder_global_map_and_freq_bias.py`.
- **Tied vs untied head:** **Press & Wolf 2017** (arXiv:1608.05859) show output and input embeddings
  can be tied; SALT3 instead trains an **untied** head built by a global emb→dec map, but the
  frequency *bias* is the part that carries the marginal — complementary to the weight tying choice.
- **Frequency structure of the head:** **adaptive softmax** (Grave et al. 2016, arXiv:1609.04309)
  exploits the same Zipfian skew for efficiency, evidence that frequency is first-class structure in
  the output layer.

## Results / why it matters
A log-frequency output bias is a cheap, well-grounded way to make a freshly re-initialized LM head
non-pathological at step 0. In SALT3's diagnostics, the random-init floor is `log(V)` (uniform);
the freq-bias head starts well below that, giving CPT a better starting point and avoiding a large
warmup spike on a brand-new vocabulary.

## How NeoBERT / SALT3 uses this
This is a **SALT3-specific improvement over origin SALT/FOCUS**. The decoder is built as
`W = E_salt @ M` (global NeoBERT emb→dec map) **plus** a bias `b = Vietnamese unigram log-frequency`.
`scripts/test_decoder_global_map_and_freq_bias.py` validates both fixes on identical hidden states so
the gain is **head-only**; `salt3_decoder_variants.py` keeps this bias **identical across decoder
arms** so weight-construction comparisons are clean. The diagnostics track `SALT decoder bias` norm
(the prior note "norm=229" refers to an earlier zeroed/garbled bias being replaced by this principled
log-freq bias).

## Relation: [[salt]] [[focus]] [[neobert]] [[culturax]]
