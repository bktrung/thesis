"""Measure how much crosslingual signal a SALT init artifact ACTUALLY carries.

Answers "is the pipeline using our init power, or just NeoBERT + data?" without
any training, by probing the SAVED artifact directly:

  1. Translation retrieval on NON-anchor probe pairs: for the projected Vietnamese
     embedding row, rank the true English NeoBERT row among all ~30k source rows
     by cosine. A random init ranks the gold word uniformly (median ~ vocab/2,
     acc@10 ~ 0.03%). Any real SALT signal shows up as a dramatically lower
     median rank. Anchor pairs are excluded (they are verbatim copies = trivial).
  2. Empirical random-control floor on the same probe pairs.
  3. Step-0 MLM loss decomposition (full / bias-only / weights-only) to show how
     much of the health-gate loss is the frequency-prior bias vs the embeddings.
  4. Anchor audit: identity pairs and pandas-NA-corruptible tokens in the CSVs.

Usage (Colab, after drive mount, with code dir on sys.path):
    from salt3_init_signal_probe import run_probe
    run_probe(PROJECT_ROOT / 'init' / 'videberta_salt_init_v5_globalmap_freqbias')
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

# (vi word, [acceptable English golds]) — common, mostly unambiguous, single-token.
PROBE_PAIRS = [
    ("nước", ["water"]), ("ngày", ["day"]), ("đêm", ["night"]), ("sách", ["book"]),
    ("chó", ["dog"]), ("mèo", ["cat"]), ("cá", ["fish"]), ("gà", ["chicken"]),
    ("nhà", ["house", "home"]), ("trường", ["school"]), ("bàn", ["table"]),
    ("ghế", ["chair"]), ("cửa", ["door"]), ("mưa", ["rain"]), ("gió", ["wind"]),
    ("lửa", ["fire"]), ("biển", ["sea"]), ("sông", ["river"]), ("núi", ["mountain"]),
    ("cây", ["tree"]), ("hoa", ["flower"]), ("lá", ["leaf"]), ("máu", ["blood"]),
    ("tim", ["heart"]), ("đầu", ["head"]), ("mắt", ["eye"]), ("tai", ["ear"]),
    ("tay", ["hand", "arm"]), ("mẹ", ["mother"]), ("cha", ["father"]),
    ("vua", ["king"]), ("tiền", ["money"]), ("vàng", ["gold"]), ("bạc", ["silver"]),
    ("đá", ["stone", "rock"]), ("muối", ["salt"]), ("sữa", ["milk"]),
    ("trứng", ["egg"]), ("thịt", ["meat"]), ("cơm", ["rice"]), ("trà", ["tea"]),
    ("ăn", ["eat"]), ("uống", ["drink"]), ("ngủ", ["sleep"]), ("chạy", ["run"]),
    ("đi", ["go"]), ("ngồi", ["sit"]), ("cười", ["laugh", "smile"]),
    ("khóc", ["cry"]), ("sống", ["live"]), ("yêu", ["love"]),
    ("đẹp", ["beautiful"]), ("nhỏ", ["small"]), ("nhanh", ["fast"]),
    ("chậm", ["slow"]), ("mới", ["new"]), ("cũ", ["old"]), ("nóng", ["hot"]),
    ("lạnh", ["cold"]), ("trắng", ["white"]), ("đen", ["black"]), ("đỏ", ["red"]),
    ("tối", ["dark"]), ("mạnh", ["strong"]), ("yếu", ["weak"]), ("đảo", ["island"]),
    ("chim", ["bird"]), ("ngựa", ["horse"]), ("voi", ["elephant"]), ("hổ", ["tiger"]),
    ("rồng", ["dragon"]), ("mặt", ["face"]), ("miệng", ["mouth"]),
    ("răng", ["tooth", "teeth"]), ("tóc", ["hair"]), ("da", ["skin"]),
    ("xương", ["bone"]), ("trăng", ["moon"]), ("tuyết", ["snow"]), ("vợ", ["wife"]),
    ("chồng", ["husband"]), ("bạn", ["friend"]), ("làng", ["village"]),
    ("chợ", ["market"]), ("thuyền", ["boat"]), ("cầu", ["bridge"]),
    ("tường", ["wall"]), ("vườn", ["garden"]), ("người", ["person", "people"]),
    ("sao", ["star"]), ("trời", ["sky", "heaven"]), ("đất", ["land", "earth"]),
    ("gỗ", ["wood"]), ("giấy", ["paper"]), ("sắt", ["iron"]), ("mây", ["cloud"]),
    ("học", ["learn", "study"]), ("dạy", ["teach"]), ("mua", ["buy"]),
    ("bán", ["sell"]), ("đọc", ["read"]), ("viết", ["write"]),
    ("nghe", ["hear", "listen"]), ("thấy", ["see"]),
]


def _emb_from_safetensors(model_dir: Path, keys=("model.encoder.weight", "encoder.weight")):
    from safetensors import safe_open

    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        names = set(f.keys())
        for k in keys:
            if k in names:
                return f.get_tensor(k).float()
    raise KeyError(f"no embedding key in {model_dir}: tried {keys}")


def _vocab_lookup(vocab: dict, token: str) -> int | None:
    for form in (token, unicodedata.normalize("NFC", token), unicodedata.normalize("NFD", token)):
        if form in vocab:
            return vocab[form]
    return None


def _load_anchor_vi_tokens(init_dir: Path) -> set[str]:
    import pandas as pd

    vi = set()
    # salt_anchor_pairs.csv = the 3-tier mined pairs (the BULK of anchors). Anchor rows are
    # verbatim NeoBERT copies, so failing to exclude them fakes perfect retrieval for any
    # probe word that happens to be an anchor — the probe would "find signal" in the copies.
    for name in ("verified_shared_anchors.csv", "shared_numbers.csv",
                 "salt_anchor_pairs.csv", "verified_translation_pairs.csv"):
        p = init_dir / name
        if p.exists():
            df = pd.read_csv(p, dtype=str, keep_default_na=False)  # NA-parsing corrupts 'nan'/'null' tokens
            if "videberta_token" in df.columns:
                vi |= set(df["videberta_token"])
    return vi


def _retrieval_ranks(query_emb: torch.Tensor, source_emb: torch.Tensor, gold_ids: list[list[int]]):
    q = F.normalize(query_emb, dim=1)
    s = F.normalize(source_emb, dim=1)
    sims = q @ s.T  # (n_probe, src_vocab)
    ranks = []
    for i, golds in enumerate(gold_ids):
        order = sims[i].argsort(descending=True)
        pos = {int(t): r for r, t in enumerate(order.tolist())}
        ranks.append(1 + min(pos[g] for g in golds))
    return ranks


def _report_ranks(label: str, ranks: list[int], vocab: int):
    t = torch.tensor(ranks, dtype=torch.float32)
    acc = lambda k: float((t <= k).float().mean())
    mrr = float((1.0 / t).mean())
    print(f"  {label:18s} n={len(ranks)} acc@1={acc(1):.1%} acc@10={acc(10):.1%} "
          f"acc@100={acc(100):.1%} median_rank={int(t.median())}/{vocab} MRR={mrr:.4f}")
    return {"n": len(ranks), "acc1": acc(1), "acc10": acc(10), "acc100": acc(100),
            "median_rank": int(t.median()), "mrr": mrr}


def loss_decomposition(model_dir: Path, device: str = "cpu"):
    """Step-0 MLM loss with full head / bias-only / weights-only on 5 VI sentences."""
    from transformers import AutoTokenizer
    from salt3_common import load_model_safe

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = load_model_safe(model_dir, device=device).eval()
    sents = ["Việt Nam là một quốc gia ở Đông Nam Á.",
             "Hôm nay thời tiết rất đẹp và trời trong xanh.",
             "Kinh tế Việt Nam tăng trưởng mạnh trong năm qua.",
             "Cô ấy đọc sách trong thư viện mỗi buổi chiều.",
             "Bóng đá là môn thể thao được nhiều người yêu thích."]
    enc = tok(sents, padding=True, truncation=True, max_length=48, return_tensors="pt").to(device)
    ids = enc["input_ids"]
    torch.manual_seed(0)
    prob = torch.full(ids.shape, 0.20)
    for sid in set(tok.all_special_ids):
        prob[ids.cpu() == sid] = 0.0
    masked = torch.bernoulli(prob).bool().to(device)
    labels = torch.full_like(ids, -100)
    labels[masked] = ids[masked]
    x = ids.clone()
    x[masked] = tok.mask_token_id
    w0, b0 = model.decoder.weight.data.clone(), model.decoder.bias.data.clone()
    out = {}
    for mode in ("full", "bias_only", "weights_only"):
        model.decoder.weight.data.copy_(torch.zeros_like(w0) if mode == "bias_only" else w0)
        model.decoder.bias.data.copy_(torch.zeros_like(b0) if mode == "weights_only" else b0)
        with torch.no_grad():
            logits = model(input_ids=x, attention_mask=enc["attention_mask"]).logits
        out[mode] = float(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100))
    model.decoder.weight.data.copy_(w0)
    model.decoder.bias.data.copy_(b0)
    print("  step-0 MLM loss: " + " | ".join(f"{k}={v:.2f}" for k, v in out.items()))
    print("  (full ≈ bias_only ⇒ health-gate loss measures the freq bias, NOT the SALT embeddings)")
    return out


def run_probe(init_dir: str | Path, source_model_id: str = "chandar-lab/NeoBERT",
              loss_probe: bool = True, device: str = "cpu", seed: int = 0):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from salt3_common import load_tokenizer_no_remote_code, write_json

    init_dir = Path(init_dir)
    model_dir = init_dir / "model"
    print(f"── SALT init signal probe: {init_dir.name} ──")
    cfg = json.loads((init_dir / "salt_config.json").read_text()) if (init_dir / "salt_config.json").exists() else {}
    print(f"  decoder_init={cfg.get('decoder_init')} anchors={cfg.get('anchor_pairs')} "
          f"projection_stats={cfg.get('projection_stats')}")

    init_emb = _emb_from_safetensors(model_dir)
    tgt_vocab = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True).get_vocab()
    # hub NeoBERT routes AutoTokenizer/AutoModel through model.py (hard xformers import),
    # so use the stripped local tokenizer + raw safetensors read instead of instantiation
    src_tok = load_tokenizer_no_remote_code(source_model_id, init_dir / "neobert_tokenizer_local")
    src_vocab = src_tok.get_vocab()
    src_emb = load_file(hf_hub_download(source_model_id, "model.safetensors"))["model.encoder.weight"].float()

    anchor_vi = _load_anchor_vi_tokens(init_dir)
    probes, skipped = [], {"anchored": 0, "missing": 0}
    for vi, golds in PROBE_PAIRS:
        vi_tok = "▁" + vi
        vi_id = _vocab_lookup(tgt_vocab, vi_tok)
        gold_ids = [g for g in (_vocab_lookup(src_vocab, e) for e in golds) if g is not None]
        if vi_id is None or not gold_ids:
            skipped["missing"] += 1
        elif vi_tok in anchor_vi:
            skipped["anchored"] += 1
        else:
            probes.append((vi, vi_id, gold_ids))
    print(f"  probes usable={len(probes)} skipped: in-anchor-set={skipped['anchored']} not-in-vocab={skipped['missing']}")

    vi_ids = [p[1] for p in probes]
    gold_ids = [p[2] for p in probes]
    print("\n── Translation retrieval (non-anchor tokens, cosine rank of gold EN row) ──")
    salt_stats = _report_ranks("SALT init", _retrieval_ranks(init_emb[vi_ids], src_emb, gold_ids), src_emb.shape[0])
    g = torch.Generator().manual_seed(seed)
    rnd = F.normalize(torch.randn(len(vi_ids), init_emb.shape[1], generator=g), dim=1) * init_emb.norm(dim=1).mean()
    rand_stats = _report_ranks("random control", _retrieval_ranks(rnd, src_emb, gold_ids), src_emb.shape[0])

    per_probe = sorted(
        (r, p[0], src_tok.convert_ids_to_tokens(p[2][0]))
        for r, p in zip(_retrieval_ranks(init_emb[vi_ids], src_emb, gold_ids), probes))
    print("  best 10 :", [(v, e, r) for r, v, e in per_probe[:10]])
    print("  worst 10:", [(v, e, r) for r, v, e in per_probe[-10:]])

    results = {"init": init_dir.name, "config": cfg, "salt": salt_stats, "random_control": rand_stats,
               "per_probe_rank": [{"vi": v, "en": e, "rank": r} for r, v, e in per_probe]}
    if loss_probe:
        print("\n── Step-0 loss decomposition ──")
        results["loss_decomposition"] = loss_decomposition(model_dir, device=device)
    write_json(init_dir / "init_signal_probe.json", results)
    print(f"\nSaved: {init_dir / 'init_signal_probe.json'}")
    return results
