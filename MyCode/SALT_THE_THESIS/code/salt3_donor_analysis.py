"""Donor & tokenizer comparison metrics for the SALT pipeline (notebook 05).

Question under test: which Vietnamese donor PLM (ViDeBERTa vs PhoBERT) gives SALT
the better substrate — and is the measured init failure caused by the donor's
embedding geometry, by the soundness METRIC itself, or by the FastText
anchor-selection bridge deviating from the SALT paper's own-space selection?

Each function is one experiment that prints its own numbers; the notebook's
verdict cell aggregates them against pre-registered predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def full_word_surface(token: str, tokenizer_type: str) -> str | None:
    """Bare surface form if the token is word-initial/standalone, else None.
    videberta: sentencepiece ▁-prefix marks word starts.
    phobert:   fastBPE @@-suffix marks NON-final pieces (tail fragments slip through).
    neobert:   WordPiece ##-prefix marks continuations."""
    if tokenizer_type == "videberta":
        return token[1:] if token.startswith("▁") else None
    if tokenizer_type == "phobert":
        return None if token.endswith("@@") else token
    if tokenizer_type == "neobert":
        return token if (not token.startswith("##") and token.isalpha()) else None
    raise ValueError(tokenizer_type)


def tokenizer_fertility(tokenizer, texts: list[str]) -> dict:
    """Tokens per whitespace word + unk rate. Lower fertility = vocab fits the
    language better = fewer rows needed to cover the same text."""
    n_tok = n_word = n_unk = 0
    unk = tokenizer.unk_token_id
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        n_tok += len(ids)
        n_word += len(t.split())
        if unk is not None:
            n_unk += sum(1 for i in ids if i == unk)
    return dict(fertility=n_tok / max(n_word, 1), unk_rate=n_unk / max(n_tok, 1),
                tokens=n_tok, words=n_word)


def vocab_mass_topn(tokenizer, texts: list[str], n_top: int) -> dict:
    """What fraction of actually-produced tokens would a top-n_top frequency-pruned
    vocab cover? Directly answers 'does the 30,522 budget hurt this donor?'."""
    counts: dict[int, int] = {}
    for t in texts:
        for i in tokenizer.encode(t, add_special_tokens=False):
            counts[i] = counts.get(i, 0) + 1
    total = sum(counts.values())
    top = sorted(counts.values(), reverse=True)[:n_top]
    return dict(distinct_ids_used=len(counts), total_tokens=total,
                topn_mass=sum(top) / max(total, 1))


def vocab_composition(vocab: dict, tokenizer_type: str, ft=None) -> dict:
    """Decompose a donor vocab: how much of it is SALT-usable material?
    full_words -> clean (alphabetic incl. '_') -> ft_known (every syllable in the
    FastText vocab — the pool anchors and projections draw from)."""
    n_full = n_clean = n_ft = n_digit = n_cont = 0
    for tok in vocab:
        s = full_word_surface(tok, tokenizer_type)
        if s is None:
            n_cont += 1
            continue
        n_full += 1
        if s.replace("▁", "").isdigit():
            n_digit += 1
            continue
        if len(s) < 2 or not s.replace("_", "").isalpha():
            continue
        n_clean += 1
        if ft is not None and all(ft.get_word_id(p) >= 0 for p in s.split("_")):
            n_ft += 1
    out = dict(vocab=len(vocab), full_words=n_full, continuations=n_cont,
               clean_words=n_clean, ft_known_words=n_ft, digit_tokens=n_digit)
    print(f"  vocab {out['vocab']:>7,} | full-word tokens {n_full:,} | clean VI words {n_clean:,} "
          f"| FastText-known {n_ft:,} | digits {n_digit:,} | continuation pieces {n_cont:,}")
    return out


def anchor_surface_availability(anchor_words: list[str], vocab: dict, tokenizer_type: str) -> dict:
    """How many existing anchor WORDS (space-joined, lowercased) exist as a single
    token in this vocab? Determines anchor survival without re-mining."""
    found = 0
    for w in anchor_words:
        joined = w.replace(" ", "_")
        cands = ([f"▁{joined}"] if tokenizer_type == "videberta" else [joined])
        if any(c in vocab for c in cands):
            found += 1
    out = dict(anchors=len(anchor_words), found=found, rate=found / max(len(anchor_words), 1))
    print(f"  anchor words present as single token: {found}/{len(anchor_words)} ({out['rate']:.1%})")
    return out


def embedding_isotropy(emb: torch.Tensor, n: int = 2000, seed: int = 0) -> dict:
    """Mean pairwise cosine of sampled rows, raw and mean-centered. A high raw value
    (cone-shaped table) flattens raw-cosine kNN and would confound the soundness
    metric — centering removes the common direction."""
    rng = np.random.default_rng(seed)
    rows = emb[torch.tensor(rng.choice(emb.shape[0], size=min(n, emb.shape[0]), replace=False))].float()
    E = F.normalize(rows, dim=1)
    E_c = F.normalize(rows - rows.mean(0), dim=1)
    g, g_c = E @ E.T, E_c @ E_c.T
    off = ~torch.eye(len(E), dtype=torch.bool)
    out = dict(mean_cos_raw=float(g[off].mean()), mean_cos_centered=float(g_c[off].mean()))
    print(f"  isotropy: mean pairwise cos raw {out['mean_cos_raw']:.3f} | centered {out['mean_cos_centered']:.3f} "
          f"({'ANISOTROPIC cone — raw-cos kNN suspect' if out['mean_cos_raw'] > 0.5 else 'reasonably isotropic'})")
    return out


def anchor_coherence(donor_emb: torch.Tensor, donor_vocab: dict, ft, anchor_tokens: list[str],
                     n: int = 300, k: int = 16, seed: int = 0) -> dict:
    """SALT's load-bearing assumption in OUR pipeline: anchors selected by FASTTEXT
    similarity must also be close in DONOR space, else the local lstsq is fit over
    scattered points and reconstructs noise.

    Reports, over n sampled anchors: (a) top-k overlap between FastText-neighbors and
    donor-space-neighbors within the anchor pool; (b) mean donor cosine of the
    FastText-selected top-k vs a random-k baseline — 'selected ≈ random' means the
    selection adds nothing in the space where the math runs."""
    rng = np.random.default_rng(seed)
    toks = [t for t in anchor_tokens if t in donor_vocab]

    def ftv(tok):
        clean = tok.replace("▁", "")
        parts = clean.split("_")
        if len(parts) > 1:
            return np.mean([ft.get_word_vector(p) for p in parts], axis=0)
        return ft.get_word_vector(clean)

    E = F.normalize(torch.stack([donor_emb[donor_vocab[t]] for t in toks]).float(), dim=1)
    Fv = F.normalize(torch.tensor(np.stack([ftv(t) for t in toks]), dtype=torch.float32), dim=1)
    idx = rng.choice(len(toks), size=min(n, len(toks)), replace=False)
    overlaps, sel_cos, rnd_cos = [], [], []
    for i in idx:
        ft_sims, dn_sims = Fv @ Fv[i], E @ E[i]
        ft_top = ft_sims.topk(k + 1).indices[1:]
        dn_top = dn_sims.topk(k + 1).indices[1:]
        overlaps.append(len(set(ft_top.tolist()) & set(dn_top.tolist())) / k)
        sel_cos.append(float(dn_sims[ft_top].mean()))
        rnd_cos.append(float(dn_sims[torch.tensor(rng.choice(len(toks), size=k, replace=False))].mean()))
    out = dict(n=len(idx), k=k, pool=len(toks),
               ft_vs_donor_knn_overlap=float(np.mean(overlaps)),
               donor_cos_of_ft_selected=float(np.mean(sel_cos)),
               donor_cos_of_random=float(np.mean(rnd_cos)))
    gain = out["donor_cos_of_ft_selected"] - out["donor_cos_of_random"]
    print(f"  anchor coherence (pool {out['pool']}, k={k}): FT-vs-donor kNN overlap {out['ft_vs_donor_knn_overlap']:.3f}")
    print(f"  donor-cos of FT-selected anchors {out['donor_cos_of_ft_selected']:.3f} vs random {out['donor_cos_of_random']:.3f} "
          f"(gain {gain:+.3f} -> {'selection MEANINGFUL in donor space' if gain > 0.1 else 'selection ~RANDOM in donor space — SALT assumption violated'})")
    return out


def _sample_vi_word_tokens(donor_vocab: dict, tokenizer_type: str, ft, n: int, seed: int):
    """Sampled full-word VI tokens present in FastText, with their donor ids and vectors."""
    rng = np.random.default_rng(seed)
    cands = []
    for tok in donor_vocab:
        s = full_word_surface(tok, tokenizer_type)
        if s is None:
            continue
        s = s.replace("▁", "")
        if len(s) < 2 or not s.replace("_", "").isalpha():
            continue
        syls = s.split("_")
        if any(ft.get_word_id(x) < 0 for x in syls):
            continue
        cands.append((tok, s, syls))
    sel = [cands[i] for i in rng.choice(len(cands), size=min(n, len(cands)), replace=False)]
    ids = [donor_vocab[t] for t, _, _ in sel]
    ftv = np.stack([np.mean([ft.get_word_vector(x) for x in syls], axis=0) for _, _, syls in sel])
    return sel, ids, torch.tensor(ftv, dtype=torch.float32)


def embedding_semantic_correlation(donor_emb, donor_vocab, ft, tokenizer_type="videberta",
                                   n_words: int = 1500, n_pairs: int = 40000, seed: int = 0) -> dict:
    """Spearman correlation between donor-embedding cosine and FastText-vi cosine over
    random word pairs. MUCH more sensitive than kNN-overlap@10: detects weak-but-present
    structure that the harsh top-k metric scores as ~chance. ρ≈0 => the donor's static
    geometry is unrelated to Vietnamese semantics; ρ>0.2 => real (if diffuse) signal.
    Returns Spearman + Pearson; prints verdict."""
    from scipy.stats import pearsonr, spearmanr
    rng = np.random.default_rng(seed)
    sel, ids, ftv = _sample_vi_word_tokens(donor_vocab, tokenizer_type, ft, n_words, seed)
    E = F.normalize(donor_emb[torch.tensor(ids)].float(), dim=1)
    Fv = F.normalize(ftv, dim=1)
    a = rng.integers(0, len(ids), size=n_pairs)
    b = rng.integers(0, len(ids), size=n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    dcos = (E[a] * E[b]).sum(1).numpy()
    fcos = (Fv[a] * Fv[b]).sum(1).numpy()
    rho = float(spearmanr(dcos, fcos).statistic)
    r = float(pearsonr(dcos, fcos)[0])
    out = dict(n_words=len(ids), n_pairs=int(keep.sum()), spearman=rho, pearson=r)
    verdict = ("SEMANTIC — donor cos tracks FastText" if rho > 0.2
               else "NO SEMANTIC SIGNAL — donor cos ⟂ FastText" if rho < 0.05 else "WEAK signal")
    print(f"  semantic correlation (pairs {out['n_pairs']:,}): Spearman {rho:+.3f} | Pearson {r:+.3f} -> {verdict}")
    return out


def related_vs_random_gap(donor_emb, donor_vocab, ft, tokenizer_type="videberta",
                          n: int = 1000, topk: int = 5, seed: int = 0) -> dict:
    """Interpretable check: mean donor cosine of each word to its FastText top-k neighbors
    vs to random words. Positive gap => the donor places semantically-related words closer,
    even if kNN-overlap is low. Zero gap => no semantic locality at all."""
    sel, ids, ftv = _sample_vi_word_tokens(donor_vocab, tokenizer_type, ft, n, seed)
    E = F.normalize(donor_emb[torch.tensor(ids)].float(), dim=1)
    Fv = F.normalize(ftv, dim=1)
    rng = np.random.default_rng(seed)
    rel, rnd = [], []
    for i in range(len(ids)):
        nbr = (Fv @ Fv[i]).topk(topk + 1).indices[1:]
        rel.append(float((E[nbr] @ E[i]).mean()))
        rnd.append(float((E[torch.tensor(rng.choice(len(ids), size=topk, replace=False))] @ E[i]).mean()))
    out = dict(n=len(ids), topk=topk, related_cos=float(np.mean(rel)),
               random_cos=float(np.mean(rnd)), gap=float(np.mean(rel) - np.mean(rnd)))
    print(f"  related-vs-random donor cosine: related {out['related_cos']:.3f} | random {out['random_cos']:.3f} "
          f"| gap {out['gap']:+.3f} ({'semantic locality present' if out['gap'] > 0.03 else 'no locality'})")
    return out


def stream_culturax_sample(n_docs: int = 500, max_chars: int = 2000,
                           dataset_id: str = "uonlp/CulturaX", lang: str = "vi") -> list[str]:
    """Deterministic text sample for tokenizer metrics (first n_docs of the stream)."""
    from datasets import load_dataset
    texts = []
    for i, row in enumerate(load_dataset(dataset_id, lang, split="train", streaming=True)):
        if i >= n_docs:
            break
        t = row["text"].strip().replace("\n", " ")
        if len(t) > 50:
            texts.append(t[:max_chars])
    print(f"  CulturaX sample: {len(texts)} docs")
    return texts
