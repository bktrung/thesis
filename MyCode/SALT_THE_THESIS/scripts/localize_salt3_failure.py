#!/usr/bin/env python3
"""
Localize the SALT3 init failure in ONE run: is the bug on the EMBEDDING side
(input collapses) or the DECODER side (output head misaligned)? And which
decoder variant gives the lowest step-0 MLM loss?

This script does NOT train. It loads the SALT init, loads the original NeoBERT
as a reference, and runs targeted probes on a shared set of hidden states so the
decoder comparison is apples-to-apples (same encoder forward pass for every head).

Usage (Colab):
    !python scripts/localize_salt3_failure.py \
        /content/drive/MyDrive/SALT3/init/videberta_salt_init_v4_stable/model \
        --source chandar-lab/NeoBERT \
        --data_cache /content/drive/MyDrive/SALT3/datasets/<culturax_cache_dir> \
        --n_chunks 16

If --data_cache is omitted it falls back to a built-in Vietnamese sentence set
(noisier loss numbers, but the embedding/decoder localization still holds).

Read the final "VERDICT" section first; the sections above it are the evidence.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers (mirror salt3_common.load_model_safe to stay standalone)
# ──────────────────────────────────────────────────────────────────────────────
def load_mlm(name_or_path, device):
    """Load a NeoBERT MaskedLM via from_config + load_state_dict.

    Using from_config (not from_pretrained) avoids the HF code path that can
    re-initialize the embedding/decoder and silently overwrite SALT weights.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    name_or_path = str(name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    if Path(name_or_path).is_dir():
        st = Path(name_or_path) / "model.safetensors"
        bn = Path(name_or_path) / "pytorch_model.bin"
        state = load_file(str(st)) if st.exists() else torch.load(str(bn), map_location="cpu", weights_only=True)
    else:
        try:
            state = load_file(hf_hub_download(repo_id=name_or_path, filename="model.safetensors"))
        except Exception:
            state = torch.load(hf_hub_download(repo_id=name_or_path, filename="pytorch_model.bin"),
                               map_location="cpu", weights_only=True)

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  [load:{name_or_path}] {len(result.missing_keys)} MISSING keys "
              f"(first: {result.missing_keys[:5]})")
    if result.unexpected_keys:
        print(f"  [load:{name_or_path}] {len(result.unexpected_keys)} unexpected keys "
              f"(first: {result.unexpected_keys[:5]})")
    return model.to(device).eval(), tokenizer


def get_embedding(model):
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder.weight.detach().float()
    return model.get_input_embeddings().weight.detach().float()


def get_decoder(model):
    dec = getattr(model, "decoder", None)
    if isinstance(dec, nn.Linear):
        bias = dec.bias.detach().float() if dec.bias is not None else None
        return dec.weight.detach().float(), bias
    return None, None


def encoder_forward(model, input_ids, attention_mask):
    """Return last_hidden_state (post-RMSNorm) from the NeoBERT base encoder."""
    base = model.model if hasattr(model, "model") else model
    out = base(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


# ──────────────────────────────────────────────────────────────────────────────
# Eval batch construction
# ──────────────────────────────────────────────────────────────────────────────
VI_SENTENCES = [
    "Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á.",
    "Hôm nay thời tiết rất đẹp và trời trong xanh.",
    "Mô hình ngôn ngữ học sâu đang phát triển rất nhanh.",
    "Trường đại học Bách khoa Hà Nội là một trong những trường hàng đầu.",
    "Con mèo nhỏ đang nằm ngủ trên chiếc ghế sofa.",
    "Kinh tế Việt Nam tăng trưởng mạnh trong những năm gần đây.",
    "Phở là món ăn truyền thống nổi tiếng của người Việt.",
    "Học sinh cần chăm chỉ học tập để đạt kết quả tốt.",
    "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất nước.",
    "Bóng đá là môn thể thao được nhiều người yêu thích.",
    "Cô ấy đọc sách trong thư viện mỗi buổi chiều.",
    "Chính phủ đã ban hành nhiều chính sách hỗ trợ doanh nghiệp.",
    "Mùa thu Hà Nội đẹp với những hàng cây lá vàng rơi.",
    "Trí tuệ nhân tạo đang thay đổi cách con người làm việc.",
    "Gia đình tôi thường về quê thăm ông bà vào dịp Tết.",
    "Biển Việt Nam có nhiều bãi cát trắng và nước biển trong xanh.",
]


def build_eval_batch_from_cache(tokenizer, data_cache, n_chunks, device):
    from datasets import load_from_disk

    ds = load_from_disk(str(data_cache))
    val = ds["validation"] if "validation" in ds else ds["train"]
    n = min(n_chunks, len(val))
    rows = [val[i]["input_ids"] for i in range(n)]
    ids = torch.tensor(rows, dtype=torch.long)
    return ids.to(device)


def build_eval_batch_from_sentences(tokenizer, device, max_len=64):
    enc = tokenizer(VI_SENTENCES, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"].to(device)


def mask_full(input_ids, tokenizer, prob=0.20, seed=42):
    """100% [MASK] replacement (matches NeoBERT pretraining / FullMaskMLMCollator)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    special = set(tokenizer.all_special_ids)
    ids = input_ids.clone()
    labels = torch.full_like(ids, -100)
    probmat = torch.full(ids.shape, prob)
    for sid in special:
        probmat[ids.cpu() == sid] = 0.0
    masked = torch.bernoulli(probmat, generator=g).bool().to(ids.device)
    labels[masked] = ids[masked]
    ids[masked] = tokenizer.mask_token_id
    return ids, labels


def ce_loss(hidden, weight, bias, labels, vocab_size):
    logits = torch.matmul(hidden.float(), weight.t().to(hidden.device))
    if bias is not None:
        logits = logits + bias.to(hidden.device)
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1),
                           ignore_index=-100).item()


# ──────────────────────────────────────────────────────────────────────────────
# Collapse metrics
# ──────────────────────────────────────────────────────────────────────────────
def effective_rank(x, eps=1e-9):
    """Entropy-based effective rank of a [N, d] matrix's singular value spectrum."""
    x = x - x.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x.float())
    except Exception:
        s = torch.linalg.svdvals(x.float().cpu())
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def mean_pairwise_cosine(x, sample=512, seed=0):
    """Average pairwise cosine similarity among rows (high → collapsed)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    idx = torch.randperm(n, generator=g)[:min(sample, n)]
    v = F.normalize(x[idx].float(), dim=1)
    sim = v @ v.t()
    m = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return float(sim[m].mean())


def section(t):
    print(f"\n{'='*72}\n {t}\n{'='*72}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("init_model", help="Path to SALT init model dir (has model.safetensors + tokenizer)")
    ap.add_argument("--source", default="chandar-lab/NeoBERT", help="Original NeoBERT reference")
    ap.add_argument("--data_cache", default=None, help="Optional saved_to_disk CulturaX cache for a realistic eval batch")
    ap.add_argument("--n_chunks", type=int, default=16, help="Eval chunks to pull from --data_cache")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    section("LOAD MODELS")
    salt, tok = load_mlm(args.init_model, device)
    vocab = salt.config.vocab_size
    rand_loss = math.log(vocab)
    print(f"SALT init: {args.init_model}")
    print(f"Vocab: {vocab}  |  random-baseline loss = log(V) = {rand_loss:.3f}")

    src, src_tok = load_mlm(args.source, device)
    src_emb = get_embedding(src).cpu()          # NeoBERT input embeddings [V_s, d]
    src_dec_w, src_dec_b = get_decoder(src)     # NeoBERT trained decoder
    src_dec_w = src_dec_w.cpu()
    src_dec_b = src_dec_b.cpu() if src_dec_b is not None else None

    salt_emb = get_embedding(salt).cpu()        # SALT embeddings (already in NeoBERT space)
    salt_dec_w, salt_dec_b = get_decoder(salt)
    salt_dec_w = salt_dec_w.cpu()
    salt_dec_b = salt_dec_b.cpu() if salt_dec_b is not None else None

    emb_mean_norm = salt_emb.norm(dim=1).mean().item()
    src_dec_mean_norm = src_dec_w.norm(dim=1).mean().item()
    src_emb_mean_norm = src_emb.norm(dim=1).mean().item()

    section("0. WEIGHT STATS / SANITY")
    print(f"SALT  emb  mean-norm : {emb_mean_norm:.4f}")
    print(f"NeoBERT emb mean-norm: {src_emb_mean_norm:.4f}")
    print(f"NeoBERT dec mean-norm: {src_dec_mean_norm:.4f}  "
          f"(decoder rows live in post-RMSNorm space; usually != emb norm)")
    if salt_dec_b is not None:
        print(f"SALT decoder bias    : max|b|={salt_dec_b.abs().max():.4f}  norm={salt_dec_b.norm():.4f}")
    if src_dec_b is not None:
        print(f"NeoBERT decoder bias : max|b|={src_dec_b.abs().max():.4f}  norm={src_dec_b.norm():.4f}  "
              f"(this is what SALT zeroed out)")

    # ── Eval batch ───────────────────────────────────────────────────────────
    section("1. BUILD MASKED VIETNAMESE EVAL BATCH")
    if args.data_cache and Path(args.data_cache).exists():
        ids = build_eval_batch_from_cache(tok, args.data_cache, args.n_chunks, device)
        print(f"Eval from cache: {tuple(ids.shape)} (real CulturaX distribution)")
    else:
        ids = build_eval_batch_from_sentences(tok, device)
        print(f"Eval from built-in sentences: {tuple(ids.shape)} (no cache given)")
    masked_ids, labels = mask_full(ids, tok)
    attn = torch.ones_like(masked_ids)
    n_masked = int((labels != -100).sum())
    print(f"Masked positions: {n_masked}")

    # ── Shared encoder forward (the SAME hidden states feed every decoder) ────
    with torch.no_grad():
        h = encoder_forward(salt, masked_ids, attn)   # [B, L, d] post-RMSNorm

    # ── 2. HIDDEN-STATE COLLAPSE CHECK ───────────────────────────────────────
    section("2. HIDDEN-STATE COLLAPSE (embedding-side health)")
    h_flat = h.reshape(-1, h.shape[-1]).float().cpu()
    er = effective_rank(h_flat)
    pc = mean_pairwise_cosine(h_flat)
    print(f"SALT/VI hidden effective-rank : {er:.1f} / {h.shape[-1]}")
    print(f"SALT/VI hidden mean pairwise cos: {pc:.4f}  (→1.0 = collapsed, ~0 = diverse)")

    # English reference through ORIGINAL NeoBERT (what healthy looks like)
    eng = ["The cat sat on the warm mat near the window.",
           "Economic growth has accelerated over the past year.",
           "Deep learning models are changing how people work.",
           "She reads books in the library every afternoon.",
           "Football is a sport loved by many people worldwide.",
           "The university is one of the leading institutions here."]
    eng_enc = src_tok(eng, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    eng_ids, eng_lab = mask_full(eng_enc["input_ids"], src_tok)
    with torch.no_grad():
        eng_h = encoder_forward(src, eng_ids, torch.ones_like(eng_ids))
    eng_flat = eng_h.reshape(-1, eng_h.shape[-1]).float().cpu()
    print(f"NeoBERT/EN hidden effective-rank : {effective_rank(eng_flat):.1f} / {eng_h.shape[-1]} (reference)")
    print(f"NeoBERT/EN hidden mean pairwise cos: {mean_pairwise_cosine(eng_flat):.4f} (reference)")

    # NeoBERT's own English MLM loss = the 'this is what working looks like' number
    eng_dec_loss = ce_loss(eng_h, src_dec_w, src_dec_b, eng_lab.to(device), vocab)
    print(f"NeoBERT/EN MLM loss with its own decoder: {eng_dec_loss:.3f} "
          f"(sanity: should be low, ~2-5)")

    # ── 3. DECODER A/B/C ON THE SAME HIDDEN STATES ───────────────────────────
    section("3. DECODER VARIANTS — MLM loss on identical SALT/VI hidden states")
    results = {}

    # (A) current projected decoder
    results["projected (current)"] = ce_loss(h, salt_dec_w, salt_dec_b, labels, vocab)

    # (B) tied to projected embedding, raw
    results["tied raw (emb as head)"] = ce_loss(h, salt_emb, None, labels, vocab)

    # (C) tied, per-row renormalized to NeoBERT decoder scale
    emb_unit = salt_emb / salt_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    results["tied + per-row dec-norm"] = ce_loss(h, emb_unit * src_dec_mean_norm, None, labels, vocab)

    # (D) tied, single global scalar to dec scale
    results["tied + global scalar"] = ce_loss(
        h, salt_emb * (src_dec_mean_norm / max(emb_mean_norm, 1e-8)), None, labels, vocab)

    # (E) GLOBAL emb->dec map learned from NeoBERT itself (stable, overdetermined)
    #     Fit M: E_neo @ M ≈ D_neo over ALL NeoBERT tokens, then W = salt_emb @ M.
    M = torch.linalg.lstsq(src_emb, src_dec_w).solution           # [d, d]
    resid = (src_emb @ M - src_dec_w).norm() / src_dec_w.norm()
    print(f"  global emb→dec map relative residual on NeoBERT: {resid:.3f} "
          f"(low → a global linear head exists; high → it doesn't)")
    W_global = salt_emb @ M
    results["global emb→dec map"] = ce_loss(h, W_global, None, labels, vocab)
    # also with NeoBERT bias mean as constant offset (frequency prior proxy)
    if src_dec_b is not None:
        results["global map + dec-bias"] = ce_loss(h, W_global, src_dec_b, labels, vocab)

    # (F) random decoder floor (≈ log V)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(vocab, h.shape[-1], generator=g) * (src_dec_mean_norm / math.sqrt(h.shape[-1]))
    results["random decoder (floor)"] = ce_loss(h, rnd, None, labels, vocab)

    print(f"\n  {'variant':32s}  loss     vs random")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        flag = "  <-- best" if v == min(results.values()) else ""
        rel = "BELOW" if v < rand_loss else "above"
        print(f"  {k:32s}  {v:7.3f}  {rel} ({v - rand_loss:+.2f}){flag}")

    # ── 4. EMBEDDING CROSS-LINGUAL ALIGNMENT ─────────────────────────────────
    section("4. EMBEDDING ALIGNMENT — do VI tokens land near English meaning?")
    # For sample VI tokens, find nearest NeoBERT input-embedding neighbors and
    # decode them with the NeoBERT tokenizer. If SALT works, neighbors should be
    # semantically related English words.
    src_emb_unit = F.normalize(src_emb, dim=1)
    sample_words = ["▁kinh", "▁tế", "▁học", "▁nước", "▁người", "▁thành", "▁phố",
                    "▁bóng", "▁đá", "▁biển", "▁sách", "▁mèo"]
    for w in sample_words:
        tid = tok.convert_tokens_to_ids(w)
        if tid is None or tid >= salt_emb.shape[0] or tid == tok.unk_token_id:
            continue
        v = F.normalize(salt_emb[tid:tid + 1], dim=1)
        sims = (v @ src_emb_unit.t()).squeeze(0)
        top = sims.topk(6).indices.tolist()
        neigh = src_tok.convert_ids_to_tokens(top)
        print(f"  {w:10s} -> {neigh}  (max cos {sims.max():.3f})")

    # ── VERDICT ──────────────────────────────────────────────────────────────
    section("VERDICT")
    best_name = min(results, key=results.get)
    best_loss = results[best_name]
    healthy_hidden = (pc < 0.6 and er > 20)
    print(f"Random baseline           : {rand_loss:.3f}")
    print(f"NeoBERT/EN reference (good): {eng_dec_loss:.3f}")
    print(f"Best decoder variant      : '{best_name}' = {best_loss:.3f}")
    print(f"Hidden states healthy?    : {'YES' if healthy_hidden else 'NO'} "
          f"(pairwise-cos={pc:.3f}, eff-rank={er:.1f})")
    print()
    if not healthy_hidden:
        print("→ EMBEDDING-SIDE BUG. Hidden states are collapsed regardless of the head,")
        print("  so no decoder can recover. Fix the embedding projection first")
        print("  (lstsq min-norm + zero-variance norm calibration likely collapses")
        print("  non-anchor directions). The decoder table is secondary until this is fixed.")
    elif best_loss < rand_loss:
        print(f"→ DECODER-SIDE BUG. Hidden states are fine; swapping the head to")
        print(f"  '{best_name}' already drops below random. The current projected")
        print(f"  decoder is the misaligned component. Adopt the best variant above.")
    else:
        print("→ PROJECTION-QUALITY ISSUE. Hidden states look diverse but EVERY head")
        print("  stays above random, meaning the per-token embedding projection itself")
        print("  is semantically wrong (bad/low anchor coverage). Inspect Section 4:")
        print("  if VI tokens do NOT map to related English neighbors, the anchor set")
        print("  or the FastText neighbor selection is the root cause.")
    print()
    print("Section 4 reading: good alignment = related English words + cos > ~0.4.")
    print("Random-looking neighbors = the embedding transfer is not preserving meaning.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()
