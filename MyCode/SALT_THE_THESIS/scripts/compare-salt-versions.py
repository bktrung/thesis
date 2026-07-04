#!/usr/bin/env python3
"""Compare SALT1/2/3 model configs, tokenizer setups, and model weights."""

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

SALT1_DIR = ROOT / "output" / "salt1"
SALT2_DIR = ROOT / "output" / "salt2"
SALT3_INIT = ROOT / "output" / "salt3" / "init_x2_tied"
SALT3_RUN = ROOT / "output" / "salt3" / "tied-dec-40k"


def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def flat_keys(d, prefix=""):
    """Flatten nested dict to dot-separated keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flat_keys(v, key))
        else:
            items[key] = v
    return items


def compare_configs():
    print("=" * 70)
    print("1. MODEL CONFIG COMPARISON (config.json)")
    print("=" * 70)

    configs = {
        "salt1": load_json(SALT1_DIR / "model" / "config.json"),
        "salt2": load_json(SALT2_DIR / "model" / "config.json"),
        "salt3": load_json(SALT3_INIT / "model" / "config.json"),
    }

    # Compare top-level keys (skip deeply nested kwargs)
    important_keys = [
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "intermediate_size", "pad_token_id", "bos_token_id", "eos_token_id",
        "max_length", "tie_word_embeddings", "use_cache", "transformers_version",
        "model_type", "architectures",
    ]

    print(f"\n{'Key':<30} {'SALT1':<20} {'SALT2':<20} {'SALT3':<20}")
    print("-" * 90)
    for key in important_keys:
        vals = []
        for name in ["salt1", "salt2", "salt3"]:
            c = configs[name]
            v = c.get(key, "MISSING") if c else "N/A"
            vals.append(str(v))
        marker = " <<<" if len(set(vals)) > 1 else ""
        print(f"{key:<30} {vals[0]:<20} {vals[1]:<20} {vals[2]:<20}{marker}")


def compare_tokenizers():
    print("\n" + "=" * 70)
    print("2. TOKENIZER COMPARISON")
    print("=" * 70)

    tok_paths = {
        "salt1": SALT1_DIR / "model" / "tokenizer.json",
        "salt2": SALT2_DIR / "model" / "tokenizer.json",
        "salt3": SALT3_INIT / "model" / "tokenizer.json",
        "salt3_pruned": SALT3_INIT / "pruned_videberta_tokenizer" / "tokenizer.json",
    }

    for name, path in tok_paths.items():
        if not path.exists():
            print(f"\n{name}: FILE NOT FOUND")
            continue

        data = load_json(path)
        model_data = data.get("model", {})
        vocab = model_data.get("vocab", [])
        added = data.get("added_tokens", [])

        print(f"\n--- {name} ({path.relative_to(ROOT)}) ---")
        print(f"  Vocab size: {len(vocab)}")
        print(f"  Added tokens: {len(added)}")
        print(f"  Model type: {model_data.get('type', 'N/A')}")
        print(f"  UNK id: {model_data.get('unk_id', 'N/A')}")

        if added:
            print(f"  Added token entries:")
            for t in sorted(added, key=lambda x: x["id"]):
                print(f"    id={t['id']:>6}  content={t['content']!r:>20}  special={t.get('special', '?')}")

        # First 10 tokens
        print(f"  First 10 vocab entries:")
        for i, entry in enumerate(vocab[:10]):
            if isinstance(entry, list):
                print(f"    [{i}] piece={entry[0]!r:<20} score={entry[1]}")
            else:
                print(f"    [{i}] {entry}")

        # Check for garbage tokens (URLs, non-Vietnamese)
        garbage = []
        for i, entry in enumerate(vocab):
            piece = entry[0] if isinstance(entry, list) else entry
            clean = piece.replace("▁", "")
            if any(frag in clean.lower() for frag in [
                "http", ".com", ".org", "www.", "youtube", "watch?",
                ".php", ".html", ".aspx", "javascript", "onclick",
            ]):
                garbage.append((i, piece))

        if garbage:
            print(f"\n  ⚠ GARBAGE TOKENS FOUND: {len(garbage)}")
            for idx, piece in garbage[:20]:
                print(f"    [{idx}] {piece!r}")
            if len(garbage) > 20:
                print(f"    ... and {len(garbage) - 20} more")


def compare_tokenizer_configs():
    print("\n" + "=" * 70)
    print("3. TOKENIZER CONFIG COMPARISON (tokenizer_config.json)")
    print("=" * 70)

    paths = {
        "salt1": SALT1_DIR / "model" / "tokenizer_config.json",
        "salt2": SALT2_DIR / "model" / "tokenizer_config.json",
        "salt3": SALT3_INIT / "model" / "tokenizer_config.json",
    }

    for name, path in paths.items():
        if not path.exists():
            print(f"\n{name}: FILE NOT FOUND")
            continue

        data = load_json(path)
        print(f"\n--- {name} ---")
        for k, v in sorted(data.items()):
            if k not in ("added_tokens_decoder",):
                print(f"  {k}: {v}")

        # Show special token mapping
        atd = data.get("added_tokens_decoder", {})
        if atd:
            print(f"  added_tokens_decoder ({len(atd)} entries):")
            for tid, info in sorted(atd.items(), key=lambda x: int(x[0])):
                if int(tid) < 10 or info.get("special", False):
                    content = info.get("content", "?")
                    special = info.get("special", False)
                    print(f"    id={tid:>6}  content={content!r:<12}  special={special}")


def compare_anchors():
    print("\n" + "=" * 70)
    print("4. ANCHOR PAIRS COMPARISON")
    print("=" * 70)

    import csv

    for name, base in [("salt1", SALT1_DIR), ("salt2", SALT2_DIR), ("salt3", SALT3_INIT)]:
        print(f"\n--- {name} ---")
        for csv_name in ["shared_numbers.csv", "verified_shared_anchors.csv", "verified_translation_pairs.csv"]:
            csv_path = base / csv_name
            if csv_path.exists():
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
                print(f"  {csv_name}: {len(rows)} pairs")
                if rows:
                    cols = list(rows[0].keys())
                    print(f"    columns: {cols}")
                    print(f"    sample: {rows[0]}")
            else:
                print(f"  {csv_name}: NOT FOUND")


def check_safetensors():
    print("\n" + "=" * 70)
    print("5. SAFETENSORS / MODEL WEIGHTS CHECK")
    print("=" * 70)

    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors not installed, skipping weight check")
        return

    for name, model_dir in [
        ("salt1", SALT1_DIR / "model"),
        ("salt2", SALT2_DIR / "model"),
        ("salt3", SALT3_INIT / "model"),
    ]:
        st_path = model_dir / "model.safetensors"
        if not st_path.exists():
            print(f"\n{name}: model.safetensors NOT FOUND")
            continue

        print(f"\n--- {name} ({st_path.relative_to(ROOT)}) ---")
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"  Total keys: {len(keys)}")

            # Check key groups
            emb_keys = [k for k in keys if "encoder.weight" in k]
            dec_keys = [k for k in keys if "decoder" in k]
            layer_keys = [k for k in keys if "transformer_encoder" in k]

            print(f"  Embedding keys: {emb_keys}")
            print(f"  Decoder keys: {dec_keys}")
            print(f"  Transformer layer keys: {len(layer_keys)}")

            # Shapes
            for k in emb_keys + dec_keys:
                t = f.get_tensor(k)
                print(f"  {k}: shape={list(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}, norm_mean={t.float().norm(dim=-1).mean():.4f}")

            # Check if decoder == embedding (tied?)
            if emb_keys and dec_keys:
                emb = f.get_tensor(emb_keys[0]).float()
                for dk in dec_keys:
                    if "weight" in dk:
                        dec = f.get_tensor(dk).float()
                        if emb.shape == dec.shape:
                            diff = (emb - dec).abs().max().item()
                            print(f"  |embedding - decoder.weight| max diff: {diff:.2e}  {'TIED' if diff < 1e-5 else 'INDEPENDENT'}")
                        else:
                            print(f"  decoder shape {list(dec.shape)} != embedding shape {list(emb.shape)}")


def check_special_token_wiring():
    print("\n" + "=" * 70)
    print("6. SPECIAL TOKEN WIRING (CRITICAL)")
    print("=" * 70)

    neobert_specials = {
        "[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103,
    }
    print(f"NeoBERT original special IDs: {neobert_specials}")

    for name, model_dir in [
        ("salt1", SALT1_DIR / "model"),
        ("salt2", SALT2_DIR / "model"),
        ("salt3", SALT3_INIT / "model"),
    ]:
        tok_cfg = load_json(model_dir / "tokenizer_config.json")
        tok_json = load_json(model_dir / "tokenizer.json")
        if not tok_cfg or not tok_json:
            print(f"\n{name}: missing files")
            continue

        print(f"\n--- {name} ---")

        # Get special token IDs from tokenizer_config
        for key in ["pad_token", "unk_token", "cls_token", "sep_token", "mask_token", "bos_token", "eos_token"]:
            val = tok_cfg.get(key)
            print(f"  {key}: {val}")

        # Check added_tokens_decoder
        atd = tok_cfg.get("added_tokens_decoder", {})
        print(f"  Special added_tokens_decoder entries:")
        for tid, info in sorted(atd.items(), key=lambda x: int(x[0])):
            if info.get("special", False):
                print(f"    id={tid:>5}  content={info.get('content', '?')!r}")

        # Cross-check: what's at position 0, 1, 2, 3, 4 in the vocab?
        vocab = tok_json.get("model", {}).get("vocab", [])
        print(f"  Vocab positions 0-5:")
        for i in range(min(6, len(vocab))):
            entry = vocab[i]
            piece = entry[0] if isinstance(entry, list) else entry
            score = entry[1] if isinstance(entry, list) and len(entry) > 1 else "N/A"
            print(f"    [{i}] piece={piece!r:<12} score={score}")


def compare_salt3_init_vs_run_model():
    print("\n" + "=" * 70)
    print("7. SALT3 INIT vs TRAINING MODEL FINGERPRINTS")
    print("=" * 70)

    init_fp = load_json(SALT3_INIT / "init_embedding_fingerprint.json")
    base_fp = load_json(SALT3_RUN / "base_embedding_fingerprint.json")
    tok_fp = load_json(SALT3_RUN / "tokenizer_fingerprint.json")

    print(f"Init embedding fingerprint : {init_fp}")
    print(f"Run base embedding fp      : {base_fp}")
    print(f"Run tokenizer fp           : {tok_fp}")

    if init_fp and base_fp:
        if init_fp.get("sha256_first_rows") == base_fp.get("sha256_first_rows"):
            print("✓ Init and training model embeddings MATCH")
        else:
            print("✗ Init and training model embeddings MISMATCH!")
            for k in ["shape", "mean_norm", "std_norm", "sha256_first_rows"]:
                v1 = init_fp.get(k)
                v2 = base_fp.get(k)
                marker = " <<<" if v1 != v2 else ""
                print(f"  {k}: init={v1}  run={v2}{marker}")


if __name__ == "__main__":
    compare_configs()
    compare_tokenizers()
    compare_tokenizer_configs()
    compare_anchors()
    check_safetensors()
    check_special_token_wiring()
    compare_salt3_init_vs_run_model()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
