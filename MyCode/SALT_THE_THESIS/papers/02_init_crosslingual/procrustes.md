---
title: "Offline bilingual word vectors, orthogonal transformations and the inverted softmax (and the Orthogonal Procrustes problem)"
authors: Smith, Turban, Hamblin, Hammerla; (Procrustes solution: Schönemann 1966)
year: 2017
venue: ICLR 2017
arxiv: 1702.03859
url: https://arxiv.org/abs/1702.03859
tags: [init, crosslingual, mapping, alignment]
---

# Orthogonal Procrustes cross-lingual mapping

## Core idea
Two embedding spaces (e.g. a Vietnamese donor and NeoBERT) can be aligned by a **single linear map**.
If you constrain that map to be **orthogonal** (a rotation/reflection), you preserve distances and
dot products, which both stabilizes the mapping and lets it be solved in **closed form** from a set
of anchor pairs — the **Orthogonal Procrustes** solution. This is the alternative to a similarity-
weighted average: instead of blending donor rows, you *rotate the whole donor space* into the target.

## Key math / architecture details
- Given paired anchor matrices `X` (source/donor) and `Y` (target), solve
  `W* = argmin_{WᵀW=I} ‖XW − Y‖_F`.
- **Closed form:** SVD of `YᵀX = UΣVᵀ` ⇒ `W* = U Vᵀ`. (Schönemann 1966.)
- Orthogonality keeps the mapping isometric → donor geometry (neighborhoods, norms) is carried over
  intact, which matters when the target model expects a particular embedding scale (RMSNorm → [[rmsnorm]]).
- Smith et al. add the **inverted softmax** to combat hubness when retrieving translations, and show
  orthogonal maps generalize better than unconstrained ones.
- Relatedly, **unconstrained least squares** `M = lstsq(X, Y)` gives the best *linear* (non-orthogonal)
  map — SALT3 uses that form for the **global emb→dec** head map.

## Results / why it matters
Orthogonal Procrustes mapping is the backbone of offline bilingual lexicon induction (MUSE, VecMap)
and a standard, robust way to bridge two pretrained embedding spaces from a small anchor set.

## How NeoBERT / SALT3 uses this
SALT3 includes a **Procrustes init arm** as an alternative to the sparsemax-average init:
`procrustes_init_v6` / `phobert_procrustes_init_v6` build the target embeddings by fitting an
(orthogonal) map from donor→NeoBERT space on the mined anchors, then applying it to all donor rows
(see `compare-salt-versions.py`, the "Procrustes-from-anchors init" comment, `salt3_diagnostics.py`).
The **same anchor pairs** feed both the sparsemax-average and Procrustes arms, so the thesis can
compare *blend vs rotate* under identical supervision. The unconstrained `lstsq` variant of this idea
is what builds the **global decoder map** (`M = lstsq(E_neo, W_neo)`) → [[salt]].

## Relation: [[wechsel]] [[salt]] [[focus]] [[rmsnorm]]
