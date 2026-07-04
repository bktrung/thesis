---
title: Analyzing the Limitations of Cross-lingual Word Embedding Mappings
authors: Ormazabal, Artetxe, Labaka, Soroa, Agirre
year: 2019
venue: ACL 2019
arxiv: P19-1492 (ACL Anthology)
url: https://aclanthology.org/P19-1492/
tags: [theory, crosslingual, mapping, init-justification]
---

# Limits of cross-lingual mappings / the isomorphism assumption (Ormazabal et al.)

## Core idea
Offline cross-lingual embedding methods independently train monolingual spaces and align them with **one global linear transform**. This silently assumes the two spaces are **approximately isomorphic** (same structure up to rotation). The paper shows that assumption is **weak**: independently-trained spaces diverge, and a single global mapping cannot fully reconcile them. **Jointly-learned** spaces are more isomorphic and align better.

## Key findings / details
- Compared offline **mapping** vs an extended skip-gram that **jointly learns** both languages' spaces on parallel data.
- Joint learning yields **more isomorphic** embeddings, is **less sensitive to hubness**, and gives stronger bilingual lexicon induction.
- The gap is evidence that **a global linear map is fundamentally limited** when the spaces weren't trained together — exactly the SALT3 situation (NeoBERT-English vs ViDeBERTa-Vietnamese, two independently trained models).

## Why it matters
The canonical "global linear cross-lingual mapping has a ceiling" result. It pushes toward either joint training (not available here — the donor and target are fixed pretrained models) or **local / per-token** mappings that don't assume global isomorphism.

## How NeoBERT / SALT3 uses this
Together with anisotropy ([[ethayarajh-anisotropy]]) this is the **theoretical case for SALT3's per-token local maps**: NeoBERT and ViDeBERTa are independently trained, so their embedding spaces are *not* isomorphic; a single global donor→NeoBERT matrix (the Procrustes/global-map baseline → [[procrustes]]) is limited by this paper's result, while **per-token weighted-least-squares maps** built from each token's sparsemax-selected anchors sidestep the global-isomorphism assumption. This is the related-work paragraph that motivates "why we map per token, not globally," and why SALT3 reports per-token vs global as an ablation.

## Relation: [[ethayarajh-anisotropy]] [[representation-degeneration]] [[salt]] [[procrustes]] [[wechsel]]
