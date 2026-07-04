#!/usr/bin/env python3
"""
Fast path to the v5 fix WITHOUT re-mining anchors: take an existing SALT init
(good embeddings), rebuild ONLY the decoder weight (global emb->dec map, scaled)
and the decoder bias (Vietnamese unigram log-frequency), and save a new artifact.

The embeddings, tokenizer, and encoder are copied unchanged from the source init.

Usage (Colab):
    !python scripts/rebuild_decoder_v5_from_existing_init.py \
        /content/drive/MyDrive/SALT3/init/videberta_salt_init_v3_proj/model \
        /content/drive/MyDrive/SALT3/init/videberta_salt_init_v5_globalmap_freqbias/model \
        --source chandar-lab/NeoBERT \
        --scale 0.1 --bias_docs 20000

Then point CPT at: BASE_MODEL_REF = 'init/videberta_salt_init_v5_globalmap_freqbias/model'
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from salt3_common import (  # noqa: E402
    copy_neobert_remote_files, fit_embedding_to_decoder_map,
    target_unigram_logfreq_bias, write_json,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from localize_salt3_failure import load_mlm, get_embedding, get_decoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_init", help="existing init model dir (good embeddings)")
    ap.add_argument("out_init", help="output model dir for the v5 artifact")
    ap.add_argument("--source", default="chandar-lab/NeoBERT")
    ap.add_argument("--scale", type=float, default=0.1, help="decoder weight scale")
    ap.add_argument("--bias_docs", type=int, default=20000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_mlm(args.src_init, device)
    src, _ = load_mlm(args.source, "cpu")
    vocab = model.config.vocab_size

    salt_emb = get_embedding(model).cpu().float()          # keep as-is
    src_emb = get_embedding(src).cpu().float()
    src_dec_w, src_dec_b = get_decoder(src)
    print(f"NeoBERT decoder bias norm (was zeroed): {src_dec_b.norm():.2f}")

    # global emb->dec map applied to the existing embeddings, scaled
    M, resid = fit_embedding_to_decoder_map(src_emb, src_dec_w.float())
    print(f"global emb->dec relative residual: {resid:.3f}")
    new_dec = (salt_emb @ M) * args.scale

    # Vietnamese unigram log-frequency bias
    print(f"Counting VI unigram freq over {args.bias_docs} CulturaX docs...")
    new_bias, counts = target_unigram_logfreq_bias(tok, vocab, num_docs=args.bias_docs, lang="vi")
    print(f"  bias range [{new_bias.min():.2f}, {new_bias.max():.2f}]  tokens={int(counts.sum()):,}")

    # write into the model and save
    model.decoder = nn.Linear(model.config.hidden_size, vocab)
    model.decoder.weight.data.copy_(new_dec)
    model.decoder.bias.data.copy_(new_bias)
    model.config.tie_word_embeddings = False
    assert torch.isfinite(model.decoder.weight).all() and torch.isfinite(model.decoder.bias).all()

    out = Path(args.out_init)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    copy_neobert_remote_files(out, args.source)
    write_json(out.parent / "salt_config.json", {
        "init_name": out.parent.name,
        "derived_from": str(args.src_init),
        "decoder_init": "global_emb_to_decoder_map",
        "decoder_weight_scale": args.scale,
        "decoder_map_residual": resid,
        "decoder_bias_init": f"vietnamese_unigram_logfreq_{args.bias_docs}docs",
        "special_token_init": "copied_from_neobert_specials",
        "note": "embeddings/encoder reused unchanged from derived_from; only decoder+bias rebuilt",
    })
    print(f"Saved v5 artifact: {out}")
    print(f"CPT: BASE_MODEL_REF = 'init/{out.parent.name}/model'")


if __name__ == "__main__":
    main()
