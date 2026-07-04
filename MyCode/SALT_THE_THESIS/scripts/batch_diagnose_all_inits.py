#!/usr/bin/env python3
"""
Batch-diagnose ALL SALT3 init variants in the init/ folder.
Compares norm distributions, dominant tokens, and decoder setup across all inits.

Usage (Colab):
    !python scripts/batch_diagnose_all_inits.py /content/drive/MyDrive/SALT3/init

Prints a comparison table at the end.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return None, None, "no weights"

    result = model.load_state_dict(state_dict, strict=False)
    missing = len(result.missing_keys)
    unexpected = len(result.unexpected_keys)
    load_status = f"OK ({len(state_dict)} keys)" if not missing and not unexpected else f"missing={missing} unexpected={unexpected}"
    return model, tokenizer, load_status


def get_embedding(model):
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder.weight.detach().float()
    return model.get_input_embeddings().weight.detach().float()


def get_decoder_weight(model):
    if hasattr(model, "decoder") and isinstance(model.decoder, nn.Linear):
        return model.decoder.weight.detach().float()
    return None


def diagnose_single(init_dir):
    """Diagnose one init variant, return summary dict."""
    init_dir = Path(init_dir)
    model_dir = init_dir / "model"
    if not model_dir.exists():
        return {"name": init_dir.name, "error": "no model/ dir"}

    # Load config
    salt_cfg_path = init_dir / "salt_config.json"
    salt_cfg = json.loads(salt_cfg_path.read_text()) if salt_cfg_path.exists() else {}

    model, tokenizer, load_status = load_model_and_tokenizer(model_dir)
    if model is None:
        return {"name": init_dir.name, "error": load_status}

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    emb = get_embedding(model).cpu()
    dec = get_decoder_weight(model)
    dec = dec.cpu() if dec is not None else None

    vocab_size = emb.shape[0]
    norms = emb.norm(dim=1)

    # Decoder relationship
    if dec is not None and emb.shape == dec.shape:
        max_diff = (emb - dec).abs().max().item()
        decoder_type = "TIED" if max_diff < 1e-5 else f"INDEPENDENT (diff={max_diff:.2e})"
    elif dec is not None:
        decoder_type = f"SHAPE_MISMATCH (emb={list(emb.shape)} dec={list(dec.shape)})"
    else:
        decoder_type = "NOT_FOUND"

    # Norm stats
    norm_mean = norms.mean().item()
    norm_std = norms.std().item()
    norm_max = norms.max().item()
    norm_min = norms[norms > 0.01].min().item()  # skip padding
    high_norm_count = (norms > norm_mean * 2).sum().item()

    # Dominant token analysis
    test_texts = [
        "Xin chào Việt Nam",
        "Hôm nay thời tiết rất đẹp",
        "Kinh tế Việt Nam phát triển nhanh",
        "Con mèo ngồi trên bàn ăn cơm",
    ]
    from collections import Counter
    all_preds = []
    all_entropies = []
    for text in test_texts:
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        probs = F.softmax(logits[0], dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
        all_preds.extend(logits[0].argmax(dim=-1).cpu().tolist())
        all_entropies.append(entropy.mean().item())

    counter = Counter(all_preds)
    unique_predicted = len(counter)
    avg_entropy = sum(all_entropies) / len(all_entropies)
    max_entropy = math.log(vocab_size)
    top3 = counter.most_common(3)
    top3_str = ", ".join(
        f"{tokenizer.convert_ids_to_tokens([tid])[0]!r}({cnt}x,n={norms[tid]:.2f})"
        for tid, cnt in top3
    )

    # Mask probe
    mask_text = f"Xin chào {tokenizer.mask_token} Nam"
    enc = tokenizer(mask_text, return_tensors="pt").to(device)
    mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    mask_pred = "N/A"
    mask_prob = 0.0
    if len(mask_pos) > 0:
        pos = mask_pos[0].item()
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        probs = F.softmax(logits[0, pos], dim=-1)
        top_id = logits[0, pos].argmax().item()
        mask_pred = tokenizer.convert_ids_to_tokens([top_id])[0]
        mask_prob = probs[top_id].item()

    # Quick loss estimate on masked text
    test_text = "Kinh tế Việt Nam phát triển nhanh trong năm qua"
    enc = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    mask_count = max(1, int(seq_len * 0.2))
    torch.manual_seed(42)
    mask_indices = torch.randperm(seq_len - 2)[:mask_count] + 1
    labels = torch.full_like(input_ids, -100)
    labels[0, mask_indices] = input_ids[0, mask_indices]
    masked_input = input_ids.clone()
    masked_input[0, mask_indices] = tokenizer.mask_token_id

    with torch.no_grad():
        out = model(input_ids=masked_input, attention_mask=enc.get("attention_mask", torch.ones_like(masked_input)).to(device))
    logits = out.logits if hasattr(out, "logits") else out["logits"]
    tied_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.to(device).reshape(-1),
        ignore_index=-100,
    ).item()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": init_dir.name,
        "load_status": load_status,
        "decoder_type": decoder_type,
        "decoder_init": salt_cfg.get("decoder_init", "unknown"),
        "anchor_pairs": salt_cfg.get("anchor_pairs", "?"),
        "vocab_size": vocab_size,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "high_norm_count": high_norm_count,
        "avg_entropy": avg_entropy,
        "max_entropy": max_entropy,
        "unique_predicted": unique_predicted,
        "top3_tokens": top3_str,
        "mask_pred": mask_pred,
        "mask_prob": mask_prob,
        "tied_loss": tied_loss,
        "random_baseline": math.log(vocab_size),
    }


def main(init_root):
    init_root = Path(init_root)
    init_dirs = sorted([d for d in init_root.iterdir() if d.is_dir() and (d / "model").exists()])

    if not init_dirs:
        print(f"No init variants found in {init_root}")
        print(f"Expected: {init_root}/*/model/ directories")
        sys.exit(1)

    print(f"Found {len(init_dirs)} init variant(s) in {init_root}")
    print(f"{'='*80}\n")

    results = []
    for d in init_dirs:
        print(f"--- Diagnosing: {d.name} ---")
        r = diagnose_single(d)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}\n")
            continue
        print(f"  decoder: {r['decoder_type']} (init={r['decoder_init']})")
        print(f"  norms: mean={r['norm_mean']:.3f} std={r['norm_std']:.3f} min={r['norm_min']:.3f} max={r['norm_max']:.3f} (>{r['norm_mean']*2:.1f}: {r['high_norm_count']} tokens)")
        print(f"  entropy: {r['avg_entropy']:.2f}/{r['max_entropy']:.2f} ({r['avg_entropy']/r['max_entropy']:.0%})")
        print(f"  unique predictions: {r['unique_predicted']}")
        print(f"  top3: {r['top3_tokens']}")
        print(f"  mask 'Xin chào ___ Nam': {r['mask_pred']!r} (p={r['mask_prob']:.4f})")
        print(f"  MLM loss: {r['tied_loss']:.2f} (random={r['random_baseline']:.2f})")
        print()

    # Comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results to compare.")
        return

    header = f"{'Init Name':<35} {'Decoder':<12} {'Loss':>7} {'Rand':>7} {'Entropy%':>9} {'Uniq':>5} {'NormMax':>8} {'High#':>6} {'MaskPred':<20}"
    print(header)
    print("-" * len(header))
    for r in valid:
        dec_short = "tied" if "TIED" in r["decoder_type"] else "indep" if "INDEPENDENT" in r["decoder_type"] else "?"
        ent_pct = f"{r['avg_entropy']/r['max_entropy']*100:.0f}%"
        loss_marker = " ⚠" if r["tied_loss"] > r["random_baseline"] * 1.5 else " ✓" if r["tied_loss"] < r["random_baseline"] else ""
        print(f"{r['name']:<35} {dec_short:<12} {r['tied_loss']:>6.2f}{loss_marker} {r['random_baseline']:>6.2f} {ent_pct:>9} {r['unique_predicted']:>5} {r['norm_max']:>8.3f} {r['high_norm_count']:>6} {r['mask_pred']!r:<20}")

    print(f"\n⚠ = loss > 1.5× random baseline")
    print(f"✓ = loss < random baseline")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")
    for r in valid:
        if r["tied_loss"] > r["random_baseline"] * 1.5:
            print(f"\n{r['name']}:")
            print(f"  Loss {r['tied_loss']:.1f} >> random {r['random_baseline']:.1f}")
            print(f"  Only {r['unique_predicted']} unique tokens ever predicted (of {r['vocab_size']})")
            print(f"  {r['high_norm_count']} tokens have norm > 2× mean ({r['norm_mean']*2:.2f})")
            print(f"  Max norm {r['norm_max']:.2f} at top is {r['norm_max']/r['norm_mean']:.1f}× the mean")
            if r["high_norm_count"] > 100:
                print(f"  → Norm outliers likely dominate softmax. Fix: per-token normalization.")
        elif r["tied_loss"] < r["random_baseline"]:
            print(f"\n{r['name']}:")
            print(f"  Loss {r['tied_loss']:.1f} < random {r['random_baseline']:.1f} — GOOD init! ✓")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
