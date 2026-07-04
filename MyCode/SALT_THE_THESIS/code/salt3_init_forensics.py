"""SALT3 init forensics: turn the step-0 MLM loss into something you can dissect.

Where the diagnostics module asks "is the projection geometry good?", this module
asks "which component (embedding / decoder / bias) is costing us loss, and does the
old SALT init actually beat the new one when measured in the SAME harness?".

Designed for 04_debug_init_analysis.ipynb. Loss harness mirrors the health gate in
01_init_embeddings_salt3.ipynb (20% masking, special tokens excluded, fixed seed) so
numbers are comparable across inits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

# Fixed, diverse Vietnamese eval set — enough masked tokens for a stable loss, and
# stable across runs so two inits are judged on identical text.
FIXED_VI_SENTENCES = [
    "Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á.",
    "Hôm nay thời tiết rất đẹp và bầu trời trong xanh.",
    "Kinh tế Việt Nam tăng trưởng mạnh trong những năm gần đây.",
    "Cô ấy thường đọc sách trong thư viện vào mỗi buổi chiều.",
    "Bóng đá là môn thể thao được rất nhiều người yêu thích.",
    "Hà Nội là thủ đô ngàn năm văn hiến của đất nước.",
    "Học sinh cần chăm chỉ học tập để đạt kết quả tốt.",
    "Món phở bò là một đặc sản nổi tiếng của ẩm thực Việt.",
    "Công nghệ trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc.",
    "Gia đình là nơi mỗi người tìm thấy sự bình yên và yêu thương.",
    "Biển miền Trung có những bãi cát trắng trải dài tuyệt đẹp.",
    "Người nông dân làm việc vất vả trên cánh đồng lúa mỗi ngày.",
    "Chính phủ đã ban hành nhiều chính sách hỗ trợ doanh nghiệp nhỏ.",
    "Trẻ em cần được tiêm phòng đầy đủ để phòng tránh bệnh tật.",
    "Âm nhạc truyền thống mang đậm bản sắc văn hóa dân tộc.",
    "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất cả nước.",
    "Việc bảo vệ môi trường là trách nhiệm của toàn xã hội.",
    "Sinh viên đại học phải hoàn thành khóa luận tốt nghiệp.",
    "Giao thông ở các đô thị lớn thường xuyên bị ùn tắc vào giờ cao điểm.",
    "Du lịch sinh thái đang ngày càng thu hút khách quốc tế.",
]


def build_eval_sentences(tokenizer=None, n_stream: int = 0, dataset_id: str = "uonlp/CulturaX", lang: str = "vi",
                         fixed_file=None):
    """FIXED set, plus the frozen held-out file when present (decisive measurements
    all share it), plus optional streamed CulturaX-vi snippets.

    fixed_file: path to fixed_eval_sentences.json written by build_frozen_eval_file.
    Missing file degrades gracefully to the built-in 20 sentences so older runs
    keep working — but decisive (gate) numbers must come from runs WITH the file.
    """
    import json
    sents = list(FIXED_VI_SENTENCES)
    if fixed_file is not None:
        p = Path(fixed_file)
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            sents += [it["text"] for it in payload["items"]]
            print(f"  eval set: +{len(payload['items'])} frozen snippets from {p.name}")
        else:
            print(f"  ⚠ frozen eval file missing ({p}) — built-in fixed set only; NOT decisive for gates")
    if n_stream > 0:
        from datasets import load_dataset
        stream = load_dataset(dataset_id, lang, split="train", streaming=True).take(n_stream)
        for row in stream:
            t = row["text"].strip().replace("\n", " ")
            if 40 < len(t) < 300:
                sents.append(t[:300])
    return sents


def build_frozen_eval_file(path, n_sentences: int = 500, dataset_id: str = "uonlp/CulturaX", lang: str = "vi",
                           min_chars: int = 40, max_chars: int = 300, scan_limit: int = 50000):
    """Build the frozen held-out eval set ONCE: stream CulturaX-vi from the start,
    keep the first n_sentences snippets in the length band, save sentences together
    with a doc manifest (stream index + url) in one JSON. CPT data prep must exclude
    these doc ids. Refuses to overwrite — the whole point is that it never changes."""
    import json
    from datasets import load_dataset
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists — the frozen eval set must never be rebuilt. Delete manually only "
                              "if you accept that ALL previous gate numbers become incomparable.")
    items = []
    for idx, row in enumerate(load_dataset(dataset_id, lang, split="train", streaming=True)):
        if idx >= scan_limit or len(items) >= n_sentences:
            break
        t = row["text"].strip().replace("\n", " ")
        if min_chars < len(t) < max_chars:
            items.append(dict(doc_index=idx, url=row.get("url"), text=t))
    payload = dict(dataset=dataset_id, lang=lang, min_chars=min_chars, max_chars=max_chars,
                   note="frozen held-out eval set; CPT must exclude these doc_index/url ids",
                   items=items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"frozen eval set saved -> {path} ({len(items)} snippets, doc ids in manifest)")
    return payload


def ensure_fasttext(bin_path: str):
    """Load cc.vi.300 FastText, downloading to bin_path first if missing (~7GB in RAM)."""
    import fasttext, gzip, os, shutil, urllib.request
    if not os.path.exists(bin_path):
        gz = bin_path + ".gz"
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz", gz)
        tmp = bin_path + ".tmp"
        with gzip.open(gz, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.rename(tmp, bin_path)  # atomic: an interrupted gunzip never leaves a truncated .bin
        os.remove(gz)
    return fasttext.load_model(bin_path)


def _embedding_param(model):
    """Input-embedding weight Parameter via a robust getter chain. Old NeoBERT
    model.py predates get_input_embeddings(); old SALT1 read model.model.encoder."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
    except Exception:
        pass
    for chain in (("model", "encoder"), ("encoder",)):
        obj = model
        for name in chain:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "weight"):
            return obj.weight
    raise AttributeError(f"{model.__class__.__name__}: no input embedding via get_input_embeddings() / "
                         "model.model.encoder / model.encoder — inspect the checkpoint's model.py.")


@torch.no_grad()
def masked_lm_loss(model, tokenizer, sentences, mask_prob=0.2, seed=0, device="cuda", max_length=64, batch_size=16):
    """Reproducible masked-LM cross-entropy — mirrors the 01 health gate exactly."""
    model.eval()
    special = set(tokenizer.all_special_ids)
    g = torch.Generator().manual_seed(seed)
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        ids = enc["input_ids"]
        probmat = torch.full(ids.shape, float(mask_prob))
        for sid in special:
            probmat[ids.cpu() == sid] = 0.0
        masked = torch.bernoulli(probmat, generator=g).bool().to(device)
        if masked.sum() == 0:
            continue
        labels = torch.full_like(ids, -100)
        labels[masked] = ids[masked]
        mids = ids.clone()
        mids[masked] = tokenizer.mask_token_id
        logits = model(input_ids=mids, attention_mask=enc["attention_mask"]).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="sum")
        total_loss += float(loss)
        total_tok += int(masked.sum())
    return total_loss / max(total_tok, 1)


def decoder_scale_sweep(model, tokenizer, sentences, factors=(0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0), device="cuda", **kw):
    """Multiply the decoder weight by each factor (1.0 = current init) and re-measure
    loss. factor 0 = bias-only floor. If loss keeps dropping past 1.0 you are
    decoder-starved (DECODER_WEIGHT_SCALE=0.1 is too low); if it rises, the head is
    miscalibrated and the prior should dominate."""
    w = model.decoder.weight
    orig = w.detach().clone()
    rows = []
    try:
        for f in factors:
            w.data = orig * f
            rows.append(dict(factor=f, loss=masked_lm_loss(model, tokenizer, sentences, device=device, **kw)))
    finally:
        w.data = orig  # restore even on OOM/interrupt — later cells measure this model
    df = pd.DataFrame(rows)
    print("── Decoder scale sweep (factor 1.0 = current init, 0 = bias-only) ──")
    for _, r in df.iterrows():
        print(f"  x{r.factor:<5} loss {r.loss:.3f}")
    if df.loss.isna().all():
        print("  ⚠ ALL losses NaN — the model's own forward is broken (non-finite weights?); sweep is meaningless")
        return df
    best = df.loc[df.loss.idxmin()]
    print(f"  argmin factor = {best.factor} (loss {best.loss:.3f}); bias-only = {df[df.factor==0].loss.values}")
    return df


def load_init_model(path: str, device="cuda", fallback_tokenizer=None):
    """Defensively load a saved init from any layout: a dir with config.json, a
    .../model/ subdir, or a root we must search. trust_remote_code for NeoBERT's
    custom class.

    Old-artifact quirks handled:
    - checkpoint saved WITHOUT decoder.weight/bias (tied-head era): from_pretrained
      silently random-inits the head, which scores ~random loss and slanders the
      artifact. Detected via the safetensors key list; reconstructed as a TIED head
      (decoder = embeddings, bias = 0) so the model is scored as it was designed.
    - folder has no tokenizer files: fall back to the provided tokenizer when its
      length matches the model's vocab_size.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    root = Path(path)
    cand = None
    for p in [root, root / "model"]:
        if (p / "config.json").exists():
            cand = p
            break
    if cand is None:
        hits = list(root.rglob("config.json"))
        if not hits:
            raise FileNotFoundError(f"No config.json under {root} — check the path / Drive mount.")
        cand = hits[0].parent
        print(f"  (found checkpoint at {cand})")

    model = AutoModelForMaskedLM.from_pretrained(str(cand), trust_remote_code=True).to(device).eval()

    ckpt_keys: set[str] = set()
    st = cand / "model.safetensors"
    pt = cand / "pytorch_model.bin"
    if st.exists():
        from safetensors import safe_open
        with safe_open(str(st), framework="pt") as f:
            ckpt_keys = set(f.keys())
    elif pt.exists():
        sd = torch.load(str(pt), map_location="cpu", weights_only=True)
        ckpt_keys = set(sd.keys())
        del sd
    if not ckpt_keys:
        print("  ⚠ could not read checkpoint key list (sharded/unknown format) — tied-head check SKIPPED; "
              "if this checkpoint has no decoder keys, the head is silently random-init.")
    if ckpt_keys and not any(k.endswith("decoder.weight") for k in ckpt_keys) and hasattr(model, "decoder"):
        emb_w = _embedding_param(model)
        model.decoder.weight.data = emb_w.data.clone()
        if model.decoder.bias is not None:
            model.decoder.bias.data.zero_()
        print("  ⚠ checkpoint has NO decoder keys -> reconstructed TIED head (decoder=embeddings, bias=0)")

    try:
        tok = AutoTokenizer.from_pretrained(str(cand), trust_remote_code=True)
    except Exception as e:
        if fallback_tokenizer is not None and len(fallback_tokenizer) == model.config.vocab_size:
            tok = fallback_tokenizer
            print(f"  ⚠ no usable tokenizer in folder ({type(e).__name__}) -> using fallback tokenizer "
                  f"(len {len(tok)} == model vocab)")
        else:
            raise
    # NaN forward with finite components is the classic corrupt-artifact signature —
    # name the offending tensors instead of leaving a bare 'loss = nan'.
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        print(f"  ⚠ {len(bad)} parameter tensors contain non-finite values (forward WILL give NaN):")
        for n in bad[:10]:
            print(f"      {n}")
    print(f"  loaded {cand.name}: vocab={model.config.vocab_size}, tok_len={len(tok)}")
    return model, tok


def vocab_identical(tok_a, tok_b) -> bool:
    return tok_a.get_vocab() == tok_b.get_vocab()


@torch.no_grad()
def component_swap_ablation(base_model, donor_model, tokenizer, sentences, device="cuda", **kw):
    """2x2 ablation to localize a loss gap to a component. ONLY valid when the two
    inits share an identical vocab (same token->row map); otherwise a row swap is
    meaningless. Swaps embedding / decoder weight / decoder bias from donor into base,
    measures loss, restores. Returns DataFrame of {swapped, loss}."""
    emb = _embedding_param(base_model)
    dec = base_model.decoder.weight
    bias = base_model.decoder.bias          # may be None on bias-less heads
    have_bias = bias is not None and donor_model.decoder.bias is not None
    d_emb = _embedding_param(donor_model).detach().clone()
    d_dec = donor_model.decoder.weight.detach().clone()
    d_bias = donor_model.decoder.bias.detach().clone() if have_bias else None
    o_emb, o_dec = emb.detach().clone(), dec.detach().clone()
    o_bias = bias.detach().clone() if have_bias else None

    def ev(tag):
        return dict(swapped=tag, loss=masked_lm_loss(base_model, tokenizer, sentences, device=device, **kw))

    rows = [ev("none (base)")]
    try:
        emb.data = d_emb; rows.append(ev("+donor embedding")); emb.data = o_emb
        dec.data = d_dec; rows.append(ev("+donor decoder")); dec.data = o_dec
        if have_bias:
            bias.data = d_bias; rows.append(ev("+donor bias")); bias.data = o_bias
        emb.data, dec.data = d_emb, d_dec
        if have_bias:
            bias.data = d_bias
        rows.append(ev("+donor all"))
    finally:
        emb.data, dec.data = o_emb, o_dec   # restore even on OOM/interrupt
        if have_bias:
            bias.data = o_bias

    df = pd.DataFrame(rows)
    print("── Component-swap ablation (donor rows into base) ──")
    base_loss = df.iloc[0].loss
    for _, r in df.iterrows():
        print(f"  {r.swapped:<20} loss {r.loss:.3f}  (Δ {r.loss - base_loss:+.3f})")
    print("  → the single swap with the biggest negative Δ is the component carrying the gap.")
    return df


def compare_init_candidates(candidates, tokenizer, sentences, device="cuda", knn_fns=None,
                            factors=(0.0, 0.5, 1.0, 2.0, 3.5, 5.0), fallback_tokenizer=None, **kw):
    """Bake-off: score N init candidates in the SAME harness — one comparable table.

    candidates: list of (name, item) where item is a checkpoint dir (str/Path,
    loaded via load_init_model and freed after scoring), an in-memory model (uses
    `tokenizer`), or an (model, tokenizer) tuple for models with their own vocab.
    knn_fns: optional {name: fn(model) -> float} kNN-preservation closures — the
    donor context differs per candidate, so the caller builds them.

    Columns: step0_loss, bias_only floor, ctx_gain (floor − loss; gate ≥ +0.1 nats),
    sweep_argmin (decoder scale factor at min loss), knn_preservation, dec_row_norm.
    """
    import gc
    factors = tuple(factors) if 0.0 in factors else (0.0,) + tuple(factors)
    rows = []
    for name, item in candidates:
        loaded = isinstance(item, (str, Path))
        model = None
        try:
            if loaded:
                model, tok = load_init_model(str(item), device=device,
                                             fallback_tokenizer=fallback_tokenizer or tokenizer)
            elif isinstance(item, tuple):
                model, tok = item
            else:
                model, tok = item, tokenizer
            print(f"\n━━ candidate: {name} ━━")
            loss = masked_lm_loss(model, tok, sentences, device=device, **kw)
            sweep = decoder_scale_sweep(model, tok, sentences, factors=factors, device=device, **kw)
            bias_only = float(sweep.loc[sweep.factor == 0, "loss"].iloc[0])
            dec_norm = (float(model.decoder.weight.detach().float().norm(dim=1).mean())
                        if hasattr(model, "decoder") else float("nan"))
            knn = float(knn_fns[name](model)) if knn_fns and name in knn_fns else float("nan")
            rows.append(dict(candidate=name, step0_loss=loss, bias_only=bias_only,
                             ctx_gain=bias_only - loss, sweep_argmin=float(sweep.loc[sweep.loss.idxmin(), "factor"]),
                             knn_preservation=knn, dec_row_norm=dec_norm))
        except Exception as e:
            # one broken candidate must not kill the whole gate table
            print(f"\n  ⚠ candidate {name!r} failed: {e!r} — NaN row, bake-off continues")
            rows.append(dict(candidate=name, step0_loss=float("nan"), bias_only=float("nan"),
                             ctx_gain=float("nan"), sweep_argmin=float("nan"),
                             knn_preservation=float("nan"), dec_row_norm=float("nan")))
        finally:
            if loaded and model is not None:  # free GPU before the next candidate
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print("\n── Init bake-off (gate: ctx_gain ≥ +0.100 nats below the bias-only floor) ──")
    for _, r in df.iterrows():
        knn_s = f"{r.knn_preservation:.3f}" if r.knn_preservation == r.knn_preservation else "  —  "
        print(f"  {r.candidate:<28} loss {r.step0_loss:.3f}  floor {r.bias_only:.3f}  "
              f"ctx {r.ctx_gain:+.3f} [{'PASS' if r.ctx_gain >= 0.1 else 'fail'}]  "
              f"argmin x{r.sweep_argmin:<4}  kNN {knn_s}  |dec| {r.dec_row_norm:.3f}")
    return df


def shared_token_emb_cosine(model_a, tok_a, model_b, tok_b):
    """For tokens present in BOTH vocabs, cosine between their embedding rows. Works
    even when vocabs differ. Low median => the two inits place shared tokens
    differently (geometry regression); high => embeddings agree, look at decoder."""
    va, vb = tok_a.get_vocab(), tok_b.get_vocab()
    shared = set(va) & set(vb)
    Ea = _embedding_param(model_a).detach().float().cpu()
    Eb = _embedding_param(model_b).detach().float().cpu()
    ia = torch.tensor([va[t] for t in shared])
    ib = torch.tensor([vb[t] for t in shared])
    cos = F.cosine_similarity(Ea[ia], Eb[ib])
    print(f"── Shared-token embedding cosine (n={len(shared)}) ──")
    print(f"  median {cos.median():.3f}  mean {cos.mean():.3f}  <0.3 frac {(cos<0.3).float().mean():.1%}")
    return pd.DataFrame(dict(token=list(shared), cos=cos.numpy()))
