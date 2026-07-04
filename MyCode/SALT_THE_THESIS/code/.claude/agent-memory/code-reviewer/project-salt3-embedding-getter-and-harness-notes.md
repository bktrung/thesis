---
name: salt3-embedding-getter-and-harness-notes
description: Recurring SALT3 review hazards - three divergent embedding-getter implementations, and frozen-eval-file change that shifts all step-0 loss numbers
metadata:
  type: project
---

Two recurring review hazards in the SALT3 codebase (found 2026-06-10 phase-1 harness review):

1. Embedding access has THREE implementations with different fallback chains:
   `salt3_common.extract_embedding_weight`, `salt3_init_forensics._embedding_param`,
   and bare `model.get_input_embeddings()` (still in `shared_token_emb_cosine`).
   Old NeoBERT model.py predates `get_input_embeddings` (raises NotImplementedError),
   so any bare call is a landmine on old-checkpoint paths.
   **Why:** fixes get applied to the function named in the plan but siblings in the
   same old-model code path get missed.
   **How to apply:** when reviewing old-checkpoint-handling changes, grep for
   `get_input_embeddings` across all touched call paths, not just the named function.

2. Notebook 04's EVAL_SENTS now appends ~500 frozen CulturaX snippets
   (eval/fixed_eval_sentences.json). All step-0 MLM losses from runs after this
   change are NOT comparable to historical numbers (the 7.1 floor story etc.).
   **How to apply:** when a report compares new loss numbers to pre-2026-06-10
   numbers, check which eval set each was measured on.

3. Anchor-CSV contract (notebook 01 cells 13/14 ↔ notebook 04 `_load_anchor_map`):
   the donor-token column is historically named `videberta_token` even for the
   PhoBERT donor — renaming it breaks notebook 04 silently. Also note the dedupe
   divergence: nb01 5f builds anchor_map with dict(zip(...)) = LAST occurrence wins;
   nb04 uses drop_duplicates('videberta_token') = FIRST wins. Diverges only on
   donor-token collisions (likely after remap collapses `▁x`/`x`).
   **How to apply:** when reviewing anchor-CSV producers/consumers, check both
   column names AND dedupe direction.

4. v5 identity contract: INIT_MODE='salt_lle' × DONOR='videberta' must keep
   INIT_NAME 'videberta_salt_init_v5_globalmap_freqbias' and byte-identical pruning/
   LLE/save behavior. Verified 2026-06-10 by dedent-diff against git HEAD — that
   technique (extract cells from `git show HEAD:...ipynb`, strip the 4-space wrap
   indent, unified-diff) is the fast way to prove "wrapped but unchanged" claims.

Related: [[salt3-loss34-diagnosis]] (user memory namespace, same investigation).
