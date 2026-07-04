---
title: "Representation Degeneration Problem in Training Natural Language Generation Models"
authors: Gao, He, Tan, Qin, Wang, Liu
year: 2019
venue: ICLR 2019
arxiv: 1907.12009
url: https://arxiv.org/abs/1907.12009
tags: [theory, geometry, tie-untie, init-justification]
---

# Representation degeneration (Gao et al.)

## Core idea
Training language models with **weight tying** (sharing the input embedding and output/decoder matrix) plus likelihood maximization pushes the learned word embeddings into a **degenerate narrow cone** — they lose directional diversity (anisotropy), and **low-frequency tokens** are squeezed worst. This is a direct argument about whether input and output matrices *should* be tied.

## Key findings / details
- **Degeneration:** most learned embeddings collapse toward a narrow cone; the optimization of tied softmax weights drives this.
- **Frequency effect:** rare tokens cluster more tightly into the degenerate region → their representations are least expressive.
- **Anisotropy link:** the narrow-cone phenomenon is the anisotropy measured by [[ethayarajh-anisotropy]].
- **Consequence:** tying limits expressiveness; untying the matrices (or regularizing) preserves more of the embedding capacity.

## Why it matters
Explains a real cost of weight tying and gives a principled reason some large models **untie** the head. Relevant to any method that has to initialize an output head.

## How NeoBERT / SALT3 uses this
Two load-bearing consequences for the thesis:
1. **NeoBERT keeps the LM head UNTIED** (verified in `NeoBERT-HF/model.py`: separate `nn.Linear` weight+bias). Representation degeneration is the theory for *why an untied head is reasonable* — it avoids the tied-softmax cone collapse.
2. Because the head is **its own matrix with its own geometry** (not equal to the embedding), SALT3 cannot just reuse the embedding map for it. It runs **per-token SALT a second time targeting the NeoBERT decoder rows** (`build_pertoken_decoder`), and then a **freeze-align stage** lets embedding and head settle together. This paper is the Ch.2/Ch.3 justification for treating the decoder as a first-class, separately-initialized matrix rather than a tied copy.

## Relation: [[ethayarajh-anisotropy]] [[ormazabal-mapping-limits]] [[salt]] [[freq-bias-init]] [[neobert]]
