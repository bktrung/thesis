"""Diagnostic probe: load Fsoft-AIC/videberta-base CORRECTLY and isolate why the MRC
benchmark scored it ~2 F1. Self-contained — needs only transformers + torch (+ pyvi for the
segmentation check). Run on Colab; CPU works for everything here (no training).

ViDeBERTa-base is a STANDARD `deberta-v2` checkpoint (no custom remote code), so a load failure
is NOT the suspect. The prime suspect is the fast tokenizer: DebertaV2TokenizerFast has a
well-known `return_offsets_mapping` misalignment for SentencePiece, which would corrupt the
char->token gold-answer span mapping in the QA harness -> the model trains on garbage labels
-> ~2 F1 (near-random). This probe checks, in order:
  1. model loads as plain deberta-v2 (AutoModel*, no trust_remote_code)  -> proves not a load bug
  2. encoder forward is finite + fill-mask is sane                       -> proves weights are real
  3. offset_mapping recovers the gold answer span                        -> the actual 2-F1 test
  4. add_prefix_space + PyVi word-segmentation effects on tokenization

Verdict at the end says which layer is broken.
"""
from __future__ import annotations

MODEL_ID = "Fsoft-AIC/videberta-base"

# one ViQuAD-shaped example (answer is a verbatim substring of context)
SAMPLE = {
    "question": "Thủ đô của Việt Nam là thành phố nào?",
    "context": "Hà Nội là thủ đô của nước Việt Nam, một thành phố có bề dày lịch sử lâu đời.",
    "answer": "Hà Nội",
}


def load_videberta(model_id: str = MODEL_ID, add_prefix_space: bool = True):
    """Load config+tokenizer+MLM the way the official repo does (fast tokenizer, no remote code)."""
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_id)
    print(f"[1/load] model_type={cfg.model_type}  layers={cfg.num_hidden_layers}  "
          f"hidden={cfg.hidden_size}  vocab={cfg.vocab_size}")
    assert cfg.model_type == "deberta-v2", f"expected deberta-v2, got {cfg.model_type!r}"

    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_prefix_space=add_prefix_space)
    except Exception as exc:  # add_prefix_space unsupported on some builds -> retry without
        print(f"[1/load]   add_prefix_space failed ({exc}); retrying without it")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    print(f"[1/load] tokenizer={type(tok).__name__}  is_fast={tok.is_fast}")
    assert tok.is_fast, "no fast tokenizer available (DebertaV2TokenizerFast required; slow needs spm.model)"

    mlm = AutoModelForMaskedLM.from_pretrained(model_id)
    mlm.eval()
    print("[1/load] AutoModelForMaskedLM loaded OK")
    # also confirm the class the QA harness uses loads without error
    from transformers import AutoModelForQuestionAnswering
    AutoModelForQuestionAnswering.from_pretrained(model_id)
    print("[1/load] AutoModelForQuestionAnswering loaded OK (so the 2-F1 is NOT a model-class bug)")
    return cfg, tok, mlm


def probe_fillmask(tok, mlm, text: str = "Hà Nội là thủ đô của nước [MASK] ."):
    """Fill-mask sanity: sane Vietnamese predictions => encoder weights are real, forward is finite."""
    import torch

    enc = tok(text, return_tensors="pt")
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        logits = mlm(**enc).logits
    finite = bool(torch.isfinite(logits).all())
    top = []
    if len(pos):
        top = tok.convert_ids_to_tokens(logits[0, pos[0]].topk(5).indices.tolist())
    print(f"[2/fwd] logits finite={finite}  top5 for [MASK]={top}")
    return finite and bool(top)


def probe_offsets(tok, ex: dict = SAMPLE, max_len: int = 384) -> bool:
    """THE 2-F1 test: can the gold answer char-span be recovered as a token-span via
    offset_mapping? (mirrors salt3_mrc_benchmark.build_qa_rows). If not, gold labels are
    garbage and the model can never learn -> ~2 F1."""
    q, c, ans = ex["question"], ex["context"], ex["answer"]
    char_start = c.find(ans)
    assert char_start >= 0, "answer not a substring of context (fix the sample)"
    char_end = char_start + len(ans)

    enc = tok(q, c, truncation="only_second", max_length=max_len,
              return_offsets_mapping=True, return_token_type_ids=False)
    offsets = enc["offset_mapping"]
    seq_ids = enc.sequence_ids()
    ids = enc["input_ids"]

    # (a) do context-token offsets index real substrings of c?
    ctx, suspicious = 0, 0
    for ti, (s, e) in enumerate(offsets):
        if seq_ids[ti] != 1:
            continue
        ctx += 1
        if s is None or e is None or e <= s or e > len(c):
            suspicious += 1

    # (b) recover the gold span exactly as the harness does
    start_t = end_t = None
    for ti, (s, e) in enumerate(offsets):
        if seq_ids[ti] != 1 or s is None or e is None:
            continue
        if s <= char_start < e and start_t is None:
            start_t = ti
        if s < char_end <= e:
            end_t = ti
    recovered = tok.decode(ids[start_t:end_t + 1]) if (start_t is not None and end_t is not None) else None
    ok = recovered is not None and ans.replace(" ", "") in (recovered or "").replace(" ", "")

    print(f"[3/offsets] context tokens={ctx}  suspicious_offsets={suspicious}")
    print(f"[3/offsets] gold='{ans}'  recovered='{recovered}'  (start_t={start_t}, end_t={end_t})")
    print(f"[3/offsets] >>> SPAN RECOVERABLE: {ok} "
          f"{'' if ok else '<-- BROKEN: this is the ~2 F1 cause (gold spans corrupted by bad offsets)'}")
    return ok


def probe_segmentation(tok, text: str = "Hà Nội là thủ đô của nước Việt Nam"):
    """Show fertility raw vs PyVi-segmented; ViDeBERTa's 128k vocab is word-level (PyVi-trained),
    so raw input under-uses it."""
    raw = tok.tokenize(text)
    print(f"[4/seg] RAW      ({len(raw)} toks): {raw}")
    try:
        from pyvi import ViTokenizer
        seg = ViTokenizer.tokenize(text)
        seg_t = tok.tokenize(seg)
        print(f"[4/seg] PyVi '{seg}'")
        print(f"[4/seg] PyVi-seg ({len(seg_t)} toks): {seg_t}")
        print(f"[4/seg] fertility raw={len(raw)} vs seg={len(seg_t)} "
              f"(lower seg => word-level vocab actually used)")
    except ImportError:
        print("[4/seg] pyvi not installed -> `pip install pyvi` to compare word-segmented tokenization")


def _load_state_dict(model_id: str):
    """Fetch the checkpoint state_dict (safetensors preferred, else .bin)."""
    from huggingface_hub import hf_hub_download
    try:
        from safetensors.torch import load_file
        return load_file(hf_hub_download(model_id, "model.safetensors"))
    except Exception:
        import torch
        return torch.load(hf_hub_download(model_id, "pytorch_model.bin"), map_location="cpu")


def diagnose_embedding_keys(model_id: str = MODEL_ID) -> dict:
    """THE decisive check. ViDeBERTa stores word embeddings as '...word_embeddings._weight'
    (a DeBERTa-v3/GDES artifact). Standard HF expects '...word_embeddings.weight', so plain
    from_pretrained leaves the embeddings RANDOM -> garbage encoder -> ~2 F1. Confirm that, and
    that a key remap fixes it."""
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM

    sd = _load_state_dict(model_id)
    emb_keys = [k for k in sd if "word_embeddings" in k]
    ckpt_emb = next((sd[k] for k in emb_keys if k.endswith(("weight", "_weight"))), None)
    print(f"[emb] checkpoint word_embeddings keys: {emb_keys}")

    std = AutoModelForMaskedLM.from_pretrained(model_id).get_input_embeddings().weight.detach()
    std_match = ckpt_emb is not None and std.shape == ckpt_emb.shape and torch.allclose(std, ckpt_emb.to(std.dtype))
    print(f"[emb] STANDARD from_pretrained: input-emb == checkpoint? {std_match}  "
          f"(model std={std.std().item():.4f}, ckpt std={ckpt_emb.std().item():.4f})")

    cfg = AutoConfig.from_pretrained(model_id)
    m2 = AutoModelForMaskedLM.from_config(cfg)
    fixed = {k.replace("word_embeddings._weight", "word_embeddings.weight"): v for k, v in sd.items()}
    m2.load_state_dict(fixed, strict=False)
    fix = m2.get_input_embeddings().weight.detach()
    fix_match = ckpt_emb is not None and torch.allclose(fix, ckpt_emb.to(fix.dtype))
    print(f"[emb] REMAP (_weight->weight): input-emb == checkpoint? {fix_match}")
    if not std_match and fix_match:
        print("[emb] >>> ROOT CAUSE CONFIRMED: standard load leaves embeddings RANDOM; remap fixes it.")
    return {"std_match": std_match, "fix_match": fix_match, "emb_keys": emb_keys}


def load_videberta_correct(model_id: str = MODEL_ID, model_cls=None):
    """Reusable FIXED loader: returns a model with word embeddings ACTUALLY loaded (remaps the
    '..._weight' checkpoint key). Use in the MRC benchmark (build_qa_model) and as a KD teacher."""
    from transformers import AutoConfig, AutoModelForMaskedLM

    model_cls = model_cls or AutoModelForMaskedLM
    model = model_cls.from_config(AutoConfig.from_pretrained(model_id))
    sd = _load_state_dict(model_id)
    fixed = {k.replace("word_embeddings._weight", "word_embeddings.weight"): v for k, v in sd.items()}
    missing, _ = model.load_state_dict(fixed, strict=False)
    still = [k for k in missing if "word_embeddings" in k]
    assert not still, f"embeddings still missing after remap: {still}"
    return model


def run_all(model_id: str = MODEL_ID):
    print(f"=== ViDeBERTa load probe: {model_id} ===")
    cfg, tok, mlm = load_videberta(model_id)
    emb = diagnose_embedding_keys(model_id)   # decisive: are embeddings actually loaded?
    span_ok = probe_offsets(tok)
    probe_segmentation(tok)
    print("\n=== VERDICT ===")
    print(f"  loads as deberta-v2         : True")
    print(f"  offsets/spans OK            : {span_ok}")
    print(f"  embeddings load (standard)  : {emb['std_match']}")
    print(f"  embeddings load (remap fix) : {emb['fix_match']}")
    if not emb["std_match"] and emb["fix_match"]:
        print("  => ROOT CAUSE: word embeddings stored as '..._weight' -> standard from_pretrained leaves them "
              "RANDOM -> ~2 F1. FIX: use load_videberta_correct() (remaps '_weight'->'weight'). "
              "ViDeBERTa is fine; the benchmark loaded it wrong.")
    elif emb["std_match"]:
        print("  => Embeddings load fine under standard from_pretrained; the 2 F1 is elsewhere "
              "(segmentation PyVi vs VnCoreNLP, or the fine-tune). Run a 5-example overfit test.")
    return {"span_ok": span_ok, **emb}


if __name__ == "__main__":
    run_all()
