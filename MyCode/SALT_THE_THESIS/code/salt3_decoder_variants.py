"""Build decoder-init ablation artifacts from an existing SALT init artifact.

All variants share the source init's ENCODER (SALT per-token embeddings) verbatim;
only the decoder weight construction differs, so artifacts are derived by copying
the init's model dir and swapping decoder.weight in model.safetensors:

  - scale arms   : global-map decoder rescaled (v5 saved E@M*0.1 -> *0.5, *1.0)
  - tied arm     : decoder.weight = copy of embedding matrix (tied-at-init, trained
                   untied -- true weight sharing is avoided on purpose: shared
                   tensors are dropped from safetensors checkpoints, which is how an
                   older artifact lost its decoder keys and loaded a random head)
  - per-token arm: SALT Eq.3 local maps refit with NeoBERT DECODER rows as targets
                   (the historically-worse variant, retested cleanly post-NaN-fix)

The decoder BIAS (Vietnamese unigram log-frequency) is kept identical across arms
so comparisons isolate the weight-construction method.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ANCHOR_CSV_NAMES = ("verified_shared_anchors.csv", "shared_numbers.csv", "verified_translation_pairs.csv")


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project logits onto the probability simplex (Sparsemax), Martins & Astudillo 2016."""
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cum_sums = torch.cumsum(sorted_logits, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype).expand_as(logits)
    k_z = (1 + k * sorted_logits > cum_sums).sum(dim=dim, keepdim=True).float()
    tau = (torch.gather(cum_sums, dim, (k_z - 1).long()) - 1) / k_z
    return torch.clamp(logits - tau, min=0.0)


def read_anchor_map(init_dir: str | Path) -> dict[str, str]:
    """Master anchor map (vi token -> neo token) from the init's CSVs.

    keep_default_na=False: tokens like 'nan'/'null' are real vocabulary items and
    must not be silently converted to float NaN by pandas.
    """
    init_dir = Path(init_dir)
    frames = []
    for name in ANCHOR_CSV_NAMES:
        p = init_dir / name
        if p.exists():
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            frames.append(df[["videberta_token", "neobert_token"]].rename(
                columns={"videberta_token": "vi", "neobert_token": "neo"}))
        else:
            print(f"  ⚠ anchor csv missing: {p}")
    if not frames:
        return {}
    master = pd.concat(frames, ignore_index=True).drop_duplicates()
    return dict(zip(master["vi"], master["neo"]))


def rebuild_vocab_maps(pruned_tokenizer_dir: str | Path, full_tokenizer_json: str | Path):
    """Recover (target_vocab, new_to_old) from the saved pruned tokenizer.

    new_to_old maps pruned id -> ORIGINAL ViDeBERTa row id, needed to index the
    full donor embedding matrix. Rebuilt by matching piece strings, which is exact
    because pruning kept pieces verbatim (Unigram vocab entries are unique).
    """
    pruned = json.loads((Path(pruned_tokenizer_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    full = json.loads(Path(full_tokenizer_json).read_text(encoding="utf-8"))
    full_piece_to_old = {piece: old_id for old_id, (piece, _score) in enumerate(full["model"]["vocab"])}
    target_vocab, new_to_old = {}, {}
    for new_id, (piece, _score) in enumerate(pruned["model"]["vocab"]):
        target_vocab[piece] = new_id
        new_to_old[new_id] = full_piece_to_old[piece]
    return target_vocab, new_to_old


def special_token_pairs(target_tokenizer, source_tokenizer, target_vocab, source_vocab):
    """Mirror nb01: map special-token roles target->source (pad/unk/cls/sep/mask/bos/eos)."""
    pairs, seen = [], set()
    for key in ("pad_token", "unk_token", "cls_token", "sep_token", "mask_token", "bos_token", "eos_token"):
        t, s = getattr(target_tokenizer, key, None), getattr(source_tokenizer, key, None)
        if t is None or s is None:
            continue
        tid, sid = target_vocab.get(t), source_vocab.get(s)
        if tid is None or sid is None or tid in seen:
            continue
        pairs.append({"key": key, "target_id": tid, "source_id": sid})
        seen.add(tid)
    return pairs


def random_embedding_matrix(
    vocab_size: int,
    hidden: int,
    neo_emb: torch.Tensor,
    special_pairs: list[dict],
    pad_id: int | None,
    mode: str = "neobert_meannorm",
    seed: int = 42,
    init_range: float = 0.02,
) -> torch.Tensor:
    """Random target-vocab embeddings; special rows copied from NeoBERT, pad zeroed.

    mode='neobert_meannorm': unit-random directions rescaled to NeoBERT's mean row
        norm -> scale-matched to the SALT init, so a control built this way differs
        from SALT ONLY in semantics (random directions carry no crosslingual signal).
    mode='scratch_init_range': N(0, init_range) -- NeoBERT's own from-scratch embedding
        recipe, i.e. what resize_token_embeddings hands a brand-new vocab. NOT
        scale-matched to SALT (use for the naive-CPT reference, not the clean control).

    Special-token rows (pad/unk/cls/sep/mask/bos/eos) are structural plumbing, not
    Vietnamese content, so they copy NeoBERT in every arm to avoid a confound. The pad
    row is zeroed (never attended, never a target).
    """
    g = torch.Generator().manual_seed(seed)
    if mode == "neobert_meannorm":
        norms = neo_emb.float().norm(dim=1)
        mean_norm = norms[norms > 1e-6].mean().item()
        emb = F.normalize(torch.randn(vocab_size, hidden, generator=g), dim=1) * mean_norm
    elif mode == "scratch_init_range":
        emb = torch.randn(vocab_size, hidden, generator=g) * init_range
    else:
        raise ValueError(f"unknown embedding mode: {mode}")
    for p in special_pairs:
        emb[p["target_id"]] = neo_emb[p["source_id"]].float()
    if pad_id is not None and pad_id < vocab_size:
        emb[pad_id] = 0.0
    return emb


def random_decoder_matrix(
    vocab_size: int,
    hidden: int,
    decoder_init_range: float,
    special_pairs: list[dict] | None = None,
    neo_dec: torch.Tensor | None = None,
    seed: int = 43,
) -> torch.Tensor:
    """Independent random LM-head rows N(0, decoder_init_range).

    NeoBERT's from-scratch untied-head recipe: the head is initialized independently
    of the embeddings (no emb->dec map), exactly what a practitioner gets resizing the
    LM head for a new vocabulary. A different seed than the embeddings keeps the two
    uncorrelated. If special_pairs+neo_dec are given, the structural special rows copy
    NeoBERT's decoder rows so they match the other arms (not Vietnamese content).
    """
    g = torch.Generator().manual_seed(seed)
    dec = torch.randn(vocab_size, hidden, generator=g) * decoder_init_range
    if special_pairs and neo_dec is not None:
        for p in special_pairs:
            dec[p["target_id"]] = neo_dec[p["source_id"]].float()
    return dec


def build_pertoken_decoder(
    vi_emb_full: torch.Tensor,      # [V_videberta_full, h] donor word embeddings
    src_decoder: torch.Tensor,      # [V_neo_rows, h] NeoBERT decoder rows (targets)
    target_vocab: dict[str, int],
    new_to_old: dict[int, int],
    source_vocab: dict[str, int],
    anchor_map: dict[str, str],
    special_pairs: list[dict],
    ft_vector_fn,                   # token -> np.ndarray fastText vector
    device: str = "cuda",
    min_anchors: int = 8,
):
    """SALT Eq.3 per-token local maps, with NeoBERT DECODER rows as regression targets.

    Identical anchor selection to the embedding pipeline (fastText cosine ->
    sparsemax -> pinv min-norm lstsq when k >= min_anchors, sparsemax-weighted
    average of decoder rows otherwise). Anchors/specials copy source decoder rows
    directly. Returns the UNSCALED decoder weight; caller applies the scale.
    """
    h = src_decoder.shape[1]
    vi_emb = vi_emb_full.float().to(device)
    dec = src_decoder.float().to(device)
    vocab_size = len(target_vocab)
    out = torch.zeros((vocab_size, h), dtype=torch.float32, device=device)

    special_ids = {p["target_id"] for p in special_pairs}
    shared_vi = [t for t in anchor_map if t in target_vocab and target_vocab[t] not in special_ids]
    non_shared = [t for t in target_vocab if t not in anchor_map and target_vocab[t] not in special_ids]

    anchor_ft = np.stack([ft_vector_fn(t) for t in shared_vi]).astype(np.float32)
    anchor_ft_t = F.normalize(torch.tensor(anchor_ft, device=device), p=2, dim=1)
    anchor_old_rows = torch.tensor([new_to_old[target_vocab[t]] for t in shared_vi], device=device)
    anchor_dec_rows = torch.stack([dec[source_vocab[anchor_map[t]]] for t in shared_vi])

    dec_mean, dec_std = dec.mean(), dec.std()
    n_lstsq = n_avg = n_rand = 0
    ks = []
    from tqdm.auto import tqdm
    for tok in tqdm(non_shared, desc="per-token decoder"):
        new_id = target_vocab[tok]
        old_id = new_to_old[new_id]
        if old_id >= vi_emb.shape[0]:
            out[new_id] = torch.normal(dec_mean, dec_std, size=(h,), device=device)
            n_rand += 1
            continue
        ft = F.normalize(torch.tensor(ft_vector_fn(tok), dtype=torch.float32, device=device).unsqueeze(0), p=2, dim=1)
        support = sparsemax(ft @ anchor_ft_t.T, dim=1).squeeze(0)
        nz = torch.nonzero(support).squeeze(1)
        if len(nz) == 0:
            out[new_id] = torch.normal(dec_mean, dec_std, size=(h,), device=device)
            n_rand += 1
            continue
        ks.append(len(nz))
        A_vi = vi_emb[anchor_old_rows[nz]]          # [k, h] donor embedding side
        D_en = anchor_dec_rows[nz]                  # [k, h] NeoBERT decoder side
        if len(nz) >= min_anchors:
            X = torch.linalg.pinv(A_vi) @ D_en      # min-norm local map emb-space -> dec-space
            out[new_id] = vi_emb[old_id] @ X
            n_lstsq += 1
        else:
            w = support[nz].unsqueeze(1)
            out[new_id] = (D_en * w).sum(0) / w.sum()
            n_avg += 1
    for tok in shared_vi:
        out[target_vocab[tok]] = dec[source_vocab[anchor_map[tok]]]
    for p in special_pairs:
        out[p["target_id"]] = dec[p["source_id"]]

    ks_t = torch.tensor(ks, dtype=torch.float32) if ks else torch.zeros(1)
    stats = {
        "n_lstsq": n_lstsq, "n_weighted_avg": n_avg, "n_random_fallback": n_rand,
        "k_mean": float(ks_t.mean()), "k_median": float(ks_t.median()),
        "row_norm_mean": float(out.norm(dim=1).mean()),
        "src_decoder_row_norm_mean": float(dec.norm(dim=1).mean()),
    }
    return out.cpu(), stats


def write_artifact(src_init_dir: str | Path, dst_init_dir: str | Path,
                   tensor_updates: dict[str, torch.Tensor], cfg_updates: dict) -> None:
    """Copy <src>/model to <dst>/model, swap the given safetensors entries, write salt_config.json.

    Copying the model dir inherits the patched model.py/rotary.py, tokenizer files,
    config and all non-swapped weights (e.g. the NeoBERT body) byte-identically.
    """
    from safetensors.torch import load_file, save_file

    src_init_dir, dst_init_dir = Path(src_init_dir), Path(dst_init_dir)
    dst_model = dst_init_dir / "model"
    dst_model.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_init_dir / "model", dst_model, dirs_exist_ok=True)

    st_path = dst_model / "model.safetensors"
    state = load_file(str(st_path))
    for key, w in tensor_updates.items():
        assert key in state, f"{key} not in saved state ({len(state)} keys)"
        assert state[key].shape == w.shape, \
            f"{key}: shape {tuple(w.shape)} != saved {tuple(state[key].shape)}"
        assert torch.isfinite(w).all(), f"non-finite tensor for {key}"
        state[key] = w.to(state[key].dtype).contiguous()
    save_file(state, str(st_path), metadata={"format": "pt"})

    cfg = {}
    src_cfg = src_init_dir / "salt_config.json"
    if src_cfg.exists():
        cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
    cfg.update(cfg_updates)
    cfg["derived_from"] = src_init_dir.name
    (dst_init_dir / "salt_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    check = load_file(str(st_path))
    for key, w in tensor_updates.items():
        diff = (check[key].float() - w.float()).abs().max().item()
        print(f"  {dst_init_dir.name}: {key} swapped, roundtrip max|Δ|={diff:.2e}")


def write_variant(src_init_dir: str | Path, dst_init_dir: str | Path,
                  decoder_weight: torch.Tensor, cfg_updates: dict) -> None:
    """Decoder-only convenience wrapper around write_artifact (used by the 01c arms)."""
    write_artifact(src_init_dir, dst_init_dir, {"decoder.weight": decoder_weight}, cfg_updates)
