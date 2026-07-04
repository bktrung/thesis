#!/usr/bin/env python3
"""
Pick the fastest-converging decoder weight scale via a SHORT CPT per scale.

For each --scale, builds the v5 head on the existing init's embeddings
(decoder = global emb->dec map * scale, bias = VI unigram log-freq), runs a short
CPT on a small data slice, and reports the eval-loss curve. Lower/faster-dropping
eval loss => better head init. Embeddings/encoder are identical across scales, so
the only variable is the decoder scale.

Usage (Colab):
    !python scripts/sweep_decoder_scale_short_cpt.py \
        /content/drive/MyDrive/SALT3/init/videberta_salt_init_v3_proj/model \
        --source chandar-lab/NeoBERT \
        --data_cache /content/drive/MyDrive/SALT3/datasets/<culturax_cache_dir> \
        --scales 0.1 0.3 0.5 1.0 \
        --num_chunks 20000 --max_steps 400 --eval_steps 50
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import salt3_common as sc  # noqa: E402
from salt3_common import (  # noqa: E402
    fit_embedding_to_decoder_map, target_unigram_logfreq_bias,
    make_mlm_collator, make_neo_mlm_trainer_class, training_args,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from localize_salt3_failure import load_mlm, get_embedding, get_decoder  # noqa: E402


def set_decoder(model, salt_emb, M, scale, bias):
    vocab = model.config.vocab_size
    dec = nn.Linear(model.config.hidden_size, vocab)
    dec.weight.data.copy_((salt_emb @ M) * scale)
    dec.bias.data.copy_(bias)
    model.decoder = dec
    model.config.tie_word_embeddings = False
    return model


def eval_curve(trainer):
    return [(h["step"], h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_init")
    ap.add_argument("--source", default="chandar-lab/NeoBERT")
    ap.add_argument("--data_cache", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--num_chunks", type=int, default=20000)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--bias_docs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    NeoMLMTrainer = make_neo_mlm_trainer_class()

    # shared, scale-independent pieces
    _, tok = load_mlm(args.base_init, "cpu")
    src, _ = load_mlm(args.source, "cpu")
    salt_emb = get_embedding(load_mlm(args.base_init, "cpu")[0]).cpu().float()
    src_emb = get_embedding(src).cpu().float()
    src_dec_w, _ = get_decoder(src)
    M, resid = fit_embedding_to_decoder_map(src_emb, src_dec_w.float())
    print(f"global emb->dec residual: {resid:.3f}")
    bias, counts = target_unigram_logfreq_bias(tok, len(tok), num_docs=args.bias_docs, lang="vi")
    print(f"VI freq bias tokens counted: {int(counts.sum()):,}")

    train_ds, eval_ds = _load_cache_directly(args.data_cache, args.num_chunks)
    collator = make_mlm_collator(tok, mlm_probability=0.20)

    summary = {}
    for scale in args.scales:
        print(f"\n{'='*60}\n SCALE {scale}\n{'='*60}")
        sc.set_seed(42)
        model, _ = load_mlm(args.base_init, device)
        model = set_decoder(model, salt_emb, M, scale, bias).to(device)
        out_dir = Path("/tmp") / f"sweep_scale_{scale}"
        args_tr = training_args(
            out_dir, max_steps=args.max_steps, learning_rate=args.lr,
            eval_steps=args.eval_steps, save_strategy="no", logging_steps=args.eval_steps,
            load_best_model_at_end=False, eval_strategy="steps",
            per_device_train_batch_size=16, gradient_accumulation_steps=4,
        )
        trainer = NeoMLMTrainer(model=model, args=args_tr, train_dataset=train_ds,
                                eval_dataset=eval_ds, data_collator=collator, processing_class=tok)
        pre = trainer.evaluate()
        trainer.train()
        post = trainer.evaluate()
        curve = eval_curve(trainer)
        summary[scale] = {"pre": pre["eval_loss"], "post": post["eval_loss"], "curve": curve}
        print(f"scale {scale}: pre={pre['eval_loss']:.3f}  post={post['eval_loss']:.3f}")
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}\n SWEEP SUMMARY (lower post = faster-converging head)\n{'='*60}")
    print(f"  {'scale':>6}  {'pre':>7}  {'post':>7}  curve")
    for scale in args.scales:
        s = summary[scale]
        cs = " ".join(f"{l:.2f}" for _, l in s["curve"])
        print(f"  {scale:6.2f}  {s['pre']:7.3f}  {s['post']:7.3f}  [{cs}]")
    best = min(summary, key=lambda k: summary[k]["post"])
    print(f"\nFastest-converging scale: {best} (post eval_loss {summary[best]['post']:.3f})")
    print("Set DECODER_WEIGHT_SCALE to this value in the init notebook.")


def _load_cache_directly(cache_dir, num_chunks):
    from datasets import load_from_disk
    ds = load_from_disk(str(cache_dir))
    train, val = ds["train"], ds["validation"]
    n_eval = max(1, int(num_chunks * 0.02))
    n_train = num_chunks - n_eval
    if n_train < len(train):
        train = train.select(range(n_train))
    if n_eval < len(val):
        val = val.select(range(n_eval))
    print(f"Using {len(train):,} train + {len(val):,} eval chunks")
    return train, val


if __name__ == "__main__":
    main()
