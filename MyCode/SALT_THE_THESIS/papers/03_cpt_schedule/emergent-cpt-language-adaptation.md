---
title: Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation
authors: Elhady, Agirre, Artetxe
year: 2025
venue: ACL 2025
arxiv: 2506.00288
url: https://arxiv.org/abs/2506.00288
tags: [cpt, crosslingual, forgetting]
---

# CPT for language adaptation — the English-replay finding

## Core idea
When you continue-pretrain an LLM to adapt it to a new language, **mixing in source-language (English) replay data is critical** — not for perplexity, but for the **emergence of downstream abilities**. Drop English and you get **catastrophic forgetting early in CPT**: validation perplexity can look fine while the model's in-context / downstream ability quietly collapses.

## Key findings / details
- **"Including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities."** → perplexity is a *misleading* monitor for language-adaptation CPT.
- Omitting English → **catastrophic forgetting early**, tied to a large shift in model parameters; the model later fails to generalize to target-language prompts despite OK perplexity.
- Proposes a **language-agnostic ICL benchmark** to detect *when* target-language abilities emerge during training.
- Alternatives to heavy English replay: **curriculum** learning and **EMA** weight averaging.
- Practical: **monitor downstream/ICL early**, not just final loss.

## Results / why it matters
Reframes language-adaptation CPT around **forgetting dynamics and capability emergence**, and warns that loss/PPL alone hide the failure mode. Directly relevant to anyone adapting a model to a new language by CPT.

## How NeoBERT / SALT3 uses this
Two uses in the thesis's CPT chapter:
1. **Justifies watching more than MLM loss** — SALT3 should report downstream (retrieval/STS/MRC) along the CPT trajectory, since this paper shows perplexity can look fine while ability degrades (it explains the "MRC plateau" nb19 chases with a fresh-data decay).
2. **Motivates a forgetting guard** — SALT3's **informed SALT init + freeze-align stage** are its way of avoiding the early catastrophic forgetting this paper documents; an English-replay / curriculum mix is the natural extension to cite as future work. → [[dont-stop-pretraining]], [[language-adaptation]], [[ibrahim-continual-pretrain]].

## Relation: [[dont-stop-pretraining]] [[language-adaptation]] [[ibrahim-continual-pretrain]] [[wsd-minicpm]] [[salt]]
