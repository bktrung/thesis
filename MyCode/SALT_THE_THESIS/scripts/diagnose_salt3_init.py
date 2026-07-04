#!/usr/bin/env python3
"""
Diagnose SALT3 init quality — run this on Colab where model weights are available.

Usage:
    python diagnose_salt3_init.py /path/to/init/model /path/to/pruned_tokenizer

Example:
    python diagnose_salt3_init.py \
        /content/drive/MyDrive/SALT3/init/videberta_salt_init_x2_tied/model \
        /content/drive/MyDrive/SALT3/init/videberta_salt_init_x2_tied/pruned_videberta_tokenizer
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_model_and_tokenizer(model_path):
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    from safetensors.torch import load_file

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    st_path = model_path / "model.safetensors"
    bin_path = model_path / "pytorch_model.bin"
    if st_path.exists():
        state_dict = load_file(str(st_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights in {model_path}")

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"WARNING: {len(result.missing_keys)} missing keys!")
        for k in result.missing_keys[:10]:
            print(f"  MISSING: {k}")
    if result.unexpected_keys:
        print(f"WARNING: {len(result.unexpected_keys)} unexpected keys!")
        for k in result.unexpected_keys[:10]:
            print(f"  UNEXPECTED: {k}")
    if not result.missing_keys and not result.unexpected_keys:
        print(f"All {len(state_dict)} keys loaded OK")

    return model, tokenizer


def get_embedding(model):
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder.weight.detach().float()
    return model.get_input_embeddings().weight.detach().float()


def get_decoder(model):
    if hasattr(model, "decoder") and isinstance(model.decoder, nn.Linear):
        return model.decoder.weight.detach().float(), model.decoder.bias.detach().float()
    return None, None


def section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def diagnose(model_path, pruned_tok_path=None):
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    emb = get_embedding(model).cpu()
    dec_w, dec_b = get_decoder(model)
    if dec_w is not None:
        dec_w = dec_w.cpu()
    if dec_b is not None:
        dec_b = dec_b.cpu()
    vocab_size = emb.shape[0]
    hidden_dim = emb.shape[1]

    section("1. EMBEDDING vs DECODER COMPARISON")
    if dec_w is not None:
        print(f"Embedding shape: {list(emb.shape)}")
        print(f"Decoder shape  : {list(dec_w.shape)}")
        max_diff = (emb - dec_w).abs().max().item()
        mean_diff = (emb - dec_w).abs().mean().item()
        cos_sim = F.cosine_similarity(emb, dec_w, dim=1)
        print(f"Max |emb - dec|    : {max_diff:.2e}")
        print(f"Mean |emb - dec|   : {mean_diff:.2e}")
        print(f"Cos similarity     : mean={cos_sim.mean():.4f}, min={cos_sim.min():.4f}, max={cos_sim.max():.4f}")
        if max_diff < 1e-5:
            print("→ TIED: decoder == embedding")
        else:
            print("→ INDEPENDENT: decoder != embedding")
    else:
        print("No decoder found")

    section("2. EMBEDDING NORM DISTRIBUTION")
    norms = emb.norm(dim=1)
    print(f"Mean norm    : {norms.mean():.4f}")
    print(f"Std norm     : {norms.std():.4f}")
    print(f"Min norm     : {norms.min():.4f} (token {norms.argmin().item()}: {tokenizer.convert_ids_to_tokens([norms.argmin().item()])[0]!r})")
    print(f"Max norm     : {norms.max():.4f} (token {norms.argmax().item()}: {tokenizer.convert_ids_to_tokens([norms.argmax().item()])[0]!r})")

    # Norm histogram
    buckets = [0, 0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 100.0]
    print("\nNorm distribution:")
    for i in range(len(buckets) - 1):
        count = ((norms >= buckets[i]) & (norms < buckets[i + 1])).sum().item()
        pct = count / vocab_size * 100
        bar = "#" * int(pct / 2)
        print(f"  [{buckets[i]:5.1f}, {buckets[i+1]:5.1f}): {count:>6} ({pct:5.1f}%) {bar}")

    # Top and bottom 10 by norm
    sorted_idx = norms.argsort()
    print("\nBottom 10 norms (excluding pad):")
    shown = 0
    for idx in sorted_idx:
        if shown >= 10:
            break
        idx = idx.item()
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        if token in ("[PAD]",):
            continue
        print(f"  [{idx:>5}] norm={norms[idx]:.4f} token={token!r}")
        shown += 1

    print("\nTop 10 norms:")
    for idx in sorted_idx[-10:].flip(0):
        idx = idx.item()
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        print(f"  [{idx:>5}] norm={norms[idx]:.4f} token={token!r}")

    section("3. DECODER BIAS CHECK")
    if dec_b is not None:
        print(f"Decoder bias range: [{dec_b.min():.4f}, {dec_b.max():.4f}]")
        print(f"Decoder bias mean : {dec_b.mean():.6f}")
        print(f"Decoder bias std  : {dec_b.std():.6f}")
        if dec_b.abs().max() > 0.1:
            print("⚠ Non-zero bias could shift predictions!")
            top_bias_idx = dec_b.argsort(descending=True)[:5]
            for idx in top_bias_idx:
                idx = idx.item()
                token = tokenizer.convert_ids_to_tokens([idx])[0]
                print(f"  [{idx}] bias={dec_b[idx]:.4f} token={token!r}")
    else:
        print("No decoder bias")

    section("4. DOMINANT TOKEN ANALYSIS")
    # Feed multiple Vietnamese texts through the model and see which tokens
    # appear most often in top-1 predictions
    test_texts = [
        "Xin chào Việt Nam",
        "Hôm nay thời tiết rất đẹp",
        "Mô hình ngôn ngữ học tiếng Việt",
        "Trường đại học Bách khoa Hà Nội",
        "Con mèo ngồi trên bàn ăn cơm",
        "Kinh tế Việt Nam phát triển nhanh",
        "Tổng thống Mỹ thăm chính thức",
        "Bóng đá Việt Nam vô địch giải đấu",
    ]

    all_preds = []
    all_entropies = []
    for text in test_texts:
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        probs = F.softmax(logits[0], dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
        pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
        all_preds.extend(pred_ids)
        all_entropies.append(entropy.mean().item())

    # Token frequency in predictions
    counter = Counter(all_preds)
    max_entropy = math.log(vocab_size)
    avg_entropy = sum(all_entropies) / len(all_entropies)
    print(f"Average entropy: {avg_entropy:.2f}/{max_entropy:.2f} ({avg_entropy/max_entropy:.0%} of max)")
    print(f"Total predictions: {len(all_preds)}")
    print(f"Unique tokens predicted: {len(counter)}")
    print(f"\nMost predicted tokens (argmax at every position):")
    for token_id, count in counter.most_common(15):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        pct = count / len(all_preds) * 100
        norm_val = norms[token_id].item()
        print(f"  [{token_id:>5}] count={count:>3} ({pct:5.1f}%) norm={norm_val:.4f} token={token!r}")

    section("5. MASKED POSITION ANALYSIS (MLM objective)")
    mask_texts = [
        f"Xin chào {tokenizer.mask_token} Nam",
        f"Hôm nay thời {tokenizer.mask_token} rất đẹp",
        f"Con {tokenizer.mask_token} ngồi trên bàn",
        f"Trường đại {tokenizer.mask_token} Bách khoa",
        f"Kinh tế {tokenizer.mask_token} Nam phát triển",
    ]

    for text in mask_texts:
        enc = tokenizer(text, return_tensors="pt").to(device)
        mask_positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
        if len(mask_positions) == 0:
            print(f"\n'{text}': NO MASK FOUND")
            continue
        pos = mask_positions[0].item()
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        mask_logits = logits[0, pos]
        probs = F.softmax(mask_logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
        top5_ids = mask_logits.topk(5).indices.cpu().tolist()
        top5_probs = probs[top5_ids].cpu().tolist()

        print(f"\n'{text}':")
        print(f"  entropy: {entropy:.2f}/{max_entropy:.2f} ({entropy/max_entropy:.0%})")
        for tid, p in zip(top5_ids, top5_probs):
            token = tokenizer.convert_ids_to_tokens([tid])[0]
            print(f"  [{tid:>5}] p={p:.4f} norm={norms[tid]:.4f} token={token!r}")

    section("6. SIMULATED RANDOM DECODER COMPARISON")
    # Compare: what loss would you get with tied decoder vs random decoder?
    # Pick one test text and compute both losses
    test_text = "Kinh tế Việt Nam phát triển nhanh trong năm qua"
    enc = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]

    # Mask 20% of tokens
    mask_count = max(1, int(seq_len * 0.2))
    torch.manual_seed(42)
    mask_indices = torch.randperm(seq_len - 2)[:mask_count] + 1  # skip CLS/SEP
    labels = torch.full_like(input_ids, -100)
    labels[0, mask_indices] = input_ids[0, mask_indices]
    masked_input = input_ids.clone()
    masked_input[0, mask_indices] = tokenizer.mask_token_id

    with torch.no_grad():
        out = model(input_ids=masked_input.to(device), attention_mask=enc.get("attention_mask", torch.ones_like(masked_input)).to(device))
    logits = out.logits if hasattr(out, "logits") else out["logits"]

    # Move labels to same device as logits for loss computation
    labels_dev = labels.to(device)

    # Loss with tied decoder (actual)
    tied_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels_dev.reshape(-1),
        ignore_index=-100,
    ).item()

    # Loss with random decoder (simulated)
    rand_decoder = torch.randn(vocab_size, hidden_dim, device=device) * 0.02
    with torch.no_grad():
        hidden = model.model(input_ids=masked_input.to(device), attention_mask=enc.get("attention_mask", torch.ones_like(masked_input)).to(device))
        if hasattr(hidden, "last_hidden_state"):
            h = hidden.last_hidden_state
        else:
            h = hidden[0] if isinstance(hidden, tuple) else hidden
    rand_logits = torch.matmul(h.float(), rand_decoder.T)
    rand_loss = F.cross_entropy(
        rand_logits.reshape(-1, vocab_size),
        labels_dev.reshape(-1),
        ignore_index=-100,
    ).item()

    # Loss with norm-equalized tied decoder (simulated fix)
    target_norm = norms.mean().item()
    eq_decoder = emb.clone().to(device)
    eq_norms = eq_decoder.norm(dim=1, keepdim=True).clamp(min=1e-8)
    eq_decoder = eq_decoder * (target_norm / eq_norms)
    eq_logits = torch.matmul(h.float(), eq_decoder.T)
    eq_loss = F.cross_entropy(
        eq_logits.reshape(-1, vocab_size),
        labels_dev.reshape(-1),
        ignore_index=-100,
    ).item()

    # Loss with original NeoBERT decoder (simulated - need to load original)
    # Skip if not available

    print(f"Test text: '{test_text}'")
    print(f"Masked positions: {mask_count}/{seq_len}")
    print(f"Random baseline (log vocab): {math.log(vocab_size):.2f}")
    print(f"Tied decoder loss          : {tied_loss:.2f}")
    print(f"Random decoder loss        : {rand_loss:.2f}")
    print(f"Norm-equalized decoder loss: {eq_loss:.2f}")
    print(f"\n  If norm-equalized << tied: norm outliers are the problem")
    print(f"  If norm-equalized ≈ tied : directional misalignment is the problem")

    section("7. ENCODER WEIGHT LOADING CHECK")
    random_looking = 0
    total_params = 0
    suspicious = []
    for name, p in model.named_parameters():
        if "transformer_encoder" in name or "encoder.layer" in name or "model.layers." in name:
            std = p.detach().float().std().item()
            total_params += 1
            if abs(std - 0.02) < 0.005 or std < 0.005:
                random_looking += 1
                suspicious.append((name, std))

    print(f"Encoder tensors: {total_params}")
    print(f"Random-looking : {random_looking}")
    if suspicious:
        print("Suspicious tensors:")
        for name, std in suspicious[:15]:
            print(f"  {name}: std={std:.6f}")

    section("8. CONFIG SANITY CHECK")
    config = model.config
    for key in ["vocab_size", "hidden_size", "num_hidden_layers", "pad_token_id",
                "bos_token_id", "eos_token_id", "tie_word_embeddings"]:
        val = getattr(config, key, "MISSING")
        marker = " ⚠" if val == "MISSING" or val is None else ""
        print(f"  {key}: {val}{marker}")

    print(f"\n  Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"  Tokenizer vocab: {len(tokenizer)}")
    print(f"  Model vocab    : {config.vocab_size}")
    if len(tokenizer) != config.vocab_size:
        print(f"  ⚠ VOCAB SIZE MISMATCH!")

    section("SUMMARY")
    print(f"Expected random loss: {math.log(vocab_size):.2f}")
    print(f"Actual tied-dec loss: {tied_loss:.2f}")
    print(f"Simulated rand loss : {rand_loss:.2f}")
    print(f"Norm-equalized loss : {eq_loss:.2f}")
    if tied_loss > math.log(vocab_size) * 2:
        print(f"\n⚠ Loss {tied_loss:.1f} >> random {math.log(vocab_size):.1f}")
        print(f"  Dominant tokens in section 4 all have norms >> mean ({norms.mean():.2f})")
        if eq_loss < tied_loss * 0.5:
            print(f"\n  FIX: Norm calibration is creating outlier embeddings.")
            print(f"  The 3.01x uniform scaling amplifies pre-existing norm")
            print(f"  variance. Tokens with high pre-scaling norms become")
            print(f"  dominant in the softmax because logit = hidden · emb,")
            print(f"  and high-norm embeddings always win.")
            print(f"\n  Proposed fix in init notebook:")
            print(f"    # Replace uniform scaling with per-token normalization")
            print(f"    target_norm = neo_mean_norm  # ~1.21")
            print(f"    cur_norms = new_embeddings[mask].norm(dim=1, keepdim=True)")
            print(f"    new_embeddings[mask] *= target_norm / cur_norms.clamp(min=1e-8)")
    if random_looking > total_params * 0.3:
        print("\n⚠ Many encoder weights look uninitialized!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    model_path = sys.argv[1]
    tok_path = sys.argv[2] if len(sys.argv) > 2 else None
    diagnose(model_path, tok_path)
