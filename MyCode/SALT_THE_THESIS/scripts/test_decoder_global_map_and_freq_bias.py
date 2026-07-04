#!/usr/bin/env python3
"""
Confirm the two proposed decoder fixes for SALT3 before changing the init notebook:
  (1) global NeoBERT emb->dec linear map as the head
  (2) Vietnamese unigram log-frequency bias (the prior SALT zeroed out, norm=229)

All variants are scored on the SAME hidden states so differences are head-only.
The Vietnamese unigram frequency is counted from the tokenized CulturaX cache.

Usage (Colab):
    !python scripts/test_decoder_global_map_and_freq_bias.py \
        /content/drive/MyDrive/SALT3/init/videberta_salt_init_v3_proj/model \
        --source chandar-lab/NeoBERT \
        --data_cache /content/drive/MyDrive/SALT3/datasets/<culturax_cache_dir> \
        --n_eval 16 --n_count 8000
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the loaders/probes from the localizer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from localize_salt3_failure import (  # noqa: E402
    load_mlm, get_embedding, get_decoder, encoder_forward, mask_full, ce_loss,
)


def vietnamese_unigram_logfreq(data_cache, vocab, n_count, device):
    """Count token frequencies over the train split and return log-frequency bias."""
    from datasets import load_from_disk

    ds = load_from_disk(str(data_cache))
    train = ds["train"] if "train" in ds else ds["validation"]
    n = min(n_count, len(train))
    counts = torch.zeros(vocab, dtype=torch.float64)
    for i in range(n):
        ids = torch.tensor(train[i]["input_ids"], dtype=torch.long)
        counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
    probs = (counts + 1.0) / (counts.sum() + vocab)   # Laplace-smoothed
    logfreq = probs.log().float()
    return logfreq.to(device), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("init_model")
    ap.add_argument("--source", default="chandar-lab/NeoBERT")
    ap.add_argument("--data_cache", required=True)
    ap.add_argument("--n_eval", type=int, default=16)
    ap.add_argument("--n_count", type=int, default=8000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    salt, tok = load_mlm(args.init_model, device)
    src, _ = load_mlm(args.source, device)
    vocab = salt.config.vocab_size

    salt_emb = get_embedding(salt).cpu()
    salt_dec_w, _ = get_decoder(salt)
    salt_dec_w = salt_dec_w.cpu()
    src_emb = get_embedding(src).cpu()
    src_dec_w, _ = get_decoder(src)
    src_dec_w = src_dec_w.cpu()

    # global emb->dec map fit on NeoBERT, applied to SALT embeddings
    M = torch.linalg.lstsq(src_emb, src_dec_w).solution
    W_global = salt_emb @ M

    # eval batch + hidden states
    from datasets import load_from_disk
    ds = load_from_disk(str(args.data_cache))
    val = ds["validation"] if "validation" in ds else ds["train"]
    ids = torch.tensor([val[i]["input_ids"] for i in range(min(args.n_eval, len(val)))],
                       dtype=torch.long).to(device)
    masked_ids, labels = mask_full(ids, tok)
    with torch.no_grad():
        h = encoder_forward(salt, masked_ids, torch.ones_like(masked_ids))

    # Vietnamese unigram frequency bias
    print(f"Counting Vietnamese unigram freq over {args.n_count} train chunks...")
    bias_vi, counts = vietnamese_unigram_logfreq(args.data_cache, vocab, args.n_count, device)
    print(f"  tokens counted: {int(counts.sum()):,}  |  bias range [{bias_vi.min():.2f}, {bias_vi.max():.2f}]")

    rand_floor = math.log(vocab)

    # frequency-only predictor (no hidden states) = best achievable from bias alone
    flat_lab = labels.reshape(-1)
    valid = flat_lab[flat_lab != -100]
    freq_only = F.cross_entropy(bias_vi.unsqueeze(0).expand(valid.shape[0], -1), valid).item()

    results = {
        "log(V) nominal floor": rand_floor,
        "freq-only (bias, no encoder)": freq_only,
        "projected current": ce_loss(h, salt_dec_w, None, labels, vocab),
        "projected + VI-freq bias": ce_loss(h, salt_dec_w, bias_vi, labels, vocab),
        "global map": ce_loss(h, W_global, None, labels, vocab),
        "global map + VI-freq bias": ce_loss(h, W_global, bias_vi, labels, vocab),
    }

    print(f"\n  {'variant':34s}  loss")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {k:34s}  {v:7.3f}")

    # Decoder-weight scaling sweep: does down-weighting the (pre-CPT harmful) encoder
    # term let the VI-freq bias dominate and approach/beat the freq-only floor?
    print(f"\n  alpha sweep: logits = alpha*(h @ W_global) + VI-freq bias")
    print(f"  {'alpha':>6s}  loss")
    best_alpha, best_alpha_loss = None, float("inf")
    for alpha in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        v = ce_loss(h, W_global * alpha, bias_vi, labels, vocab)
        marker = ""
        if v < best_alpha_loss:
            best_alpha, best_alpha_loss = alpha, v
        print(f"  {alpha:6.2f}  {v:7.3f}")
    print(f"\nBest alpha = {best_alpha} -> loss {best_alpha_loss:.3f}")
    print(f"freq-only floor = {freq_only:.3f}")
    if best_alpha_loss < freq_only - 0.05:
        print("=> Encoder IS useful when down-weighted: adopt global map * best_alpha + VI-freq bias.")
    elif best_alpha == 0.0:
        print("=> Encoder is purely harmful pre-CPT: init decoder weight ~0, rely on VI-freq bias + CPT.")
    else:
        print("=> Encoder roughly neutral when down-weighted; small alpha + VI-freq bias is the safe init.")


if __name__ == "__main__":
    main()
