"""CPU smoke checks for the MRC benchmark engine — no GPU, no network.

Validates the parts that break silently: answerable filtering, offset-aligned
span building, the word-segmentation answer-span remap, constrained-span SQuAD
decode, the NeoBERT custom QA head forward, and skip-done idempotency.
Uses an offline char/word-level fast tokenizer and stub models. Run:
    python3 code/salt3_mrc_benchmark_checks.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import salt3_mrc_benchmark as bench

_PASS, _FAIL = 0, 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {name}")
    else:
        _FAIL += 1
        print(f"  ✗ {name} {detail}")


def _build_fast_tokenizer():
    """Whitespace WordLevel fast tokenizer with a [CLS] A [SEP] B [SEP] pair template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    words = ["thủ", "đô", "của", "việt", "nam", "là", "hà", "nội", "thành", "phố",
             "thủ_đô", "việt_nam", "hà_nội", "ở", "đâu", "?", "."]
    for w in words:
        vocab[w] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, cls_token="[CLS]", sep_token="[SEP]",
                                   pad_token="[PAD]", unk_token="[UNK]")


def _examples():
    return [
        {"id": "a", "question": "thủ đô của việt nam là ?",
         "context": "thủ đô của việt nam là hà nội .",
         "answers": {"text": ["hà nội"], "answer_start": [23]}, "is_impossible": False},
        {"id": "b", "question": "hà nội ở đâu ?", "context": "hà nội là thành phố .",
         "answers": {"text": [""], "answer_start": [0]}, "is_impossible": True},  # dropped
    ]


def test_answerable_filter():
    exs = _examples()
    check("answerable example parsed", bench._first_answer(exs[0]) == ("hà nội", 23))
    check("impossible example dropped", bench._first_answer(exs[1]) is None)


def test_span_alignment_unsegmented():
    tok = _build_fast_tokenizer()
    spec = {"segment": False}
    ds = bench.build_qa_rows(_examples(), tok, spec, cap=None, max_len=32, seed=0)
    check("one answerable row kept", len(ds) == 1, f"got {len(ds)}")
    row = ds[0]
    decoded = tok.decode(row["input_ids"][row["start_positions"]:row["end_positions"] + 1])
    check("decoded gold span == 'hà nội'", decoded.replace(" ", "") == "hànội", f"got '{decoded}'")
    check("context_token_mask marks ctx tokens", sum(row["context_token_mask"]) > 0)


def test_segment_remap(monkeypatch_join=True):
    """Fake segmenter underscore-joins; remap must re-find the shifted answer span."""
    tok = _build_fast_tokenizer()
    orig = bench.maybe_segment
    bench.maybe_segment = lambda texts, spec, **k: [t.replace("hà nội", "hà_nội")
                                                    .replace("việt nam", "việt_nam")
                                                    .replace("thủ đô", "thủ_đô") for t in texts]
    try:
        spec = {"segment": True}
        ds = bench.build_qa_rows(_examples(), tok, spec, cap=None, max_len=32, seed=0)
        check("segmented row kept (span re-found)", len(ds) == 1, f"got {len(ds)}")
        if len(ds):
            decoded = tok.decode(ds[0]["input_ids"][ds[0]["start_positions"]:ds[0]["end_positions"] + 1])
            check("segmented gold span == 'hà_nội'", "hà_nội" in decoded, f"got '{decoded}'")
    finally:
        bench.maybe_segment = orig


class _SlowWordTokenizer:
    """Minimal slow (is_fast=False) word-level tokenizer mimicking PhoBERT's API surface.
    Words >3 chars split into per-char subwords to exercise multi-token-per-word offsets."""
    is_fast = False
    cls_token_id, sep_token_id, pad_token_id = 2, 3, 0

    def __init__(self):
        self._vocab, self._inv = {}, {}

    def _id(self, tok):
        if tok not in self._vocab:
            i = 4 + len(self._vocab)
            self._vocab[tok], self._inv[i] = i, tok
        return self._vocab[tok]

    def _subwords(self, word):
        return [word] if len(word) <= 3 else list(word)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            ids += [self._id(s) for s in self._subwords(w)]
        return ids

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2

    def build_inputs_with_special_tokens(self, a, b):
        return [self.cls_token_id] + a + [self.sep_token_id, self.sep_token_id] + b + [self.sep_token_id]

    def get_special_tokens_mask(self, a, b, already_has_special_tokens=False):
        return [1] + [0] * len(a) + [1, 1] + [0] * len(b) + [1]

    def decode(self, ids, skip_special_tokens=True):
        specials = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        toks = [self._inv.get(i, "") for i in ids if not (skip_special_tokens and i in specials)]
        return " ".join(t for t in toks if t)


def test_slow_wordlevel_path():
    """PhoBERT-style slow tokenizer: synthetic offsets must align + oracle F1=100."""
    tok = _SlowWordTokenizer()
    spec = {"segment": True}  # PhoBERT is always segment=True; examples already segmented
    orig = bench.maybe_segment
    bench.maybe_segment = lambda texts, spec, **k: list(texts)  # identity (already segmented)
    try:
        ds = bench.build_qa_rows(_examples(), tok, spec, cap=None, max_len=40, seed=0)
    finally:
        bench.maybe_segment = orig
    check("slow path keeps answerable row", len(ds) == 1, f"got {len(ds)}")
    if len(ds):
        row = ds[0]
        check("slow span maps to context tokens", row["context_token_mask"][row["start_positions"]] == 1)
        decoded = tok.decode(row["input_ids"][row["start_positions"]:row["end_positions"] + 1])
        check("slow decoded span == 'hà nội'", decoded == "hà nội", f"got '{decoded}'")

        gold_s, gold_e = row["start_positions"], row["end_positions"]

        class OracleModel(nn.Module):
            def forward(self, input_ids=None, attention_mask=None, **k):
                b, n = input_ids.shape
                sl, el = torch.full((b, n), -9.0), torch.full((b, n), -9.0)
                sl[:, gold_s] = 9.0
                el[:, gold_e] = 9.0
                return {"start_logits": sl, "end_logits": el}

        m = bench.squad_post_eval(OracleModel(), tok, ds, device="cpu", batch_size=4)
        check("slow oracle F1=100", abs(m["f1"] - 100.0) < 1e-6, f"f1={m['f1']}")


def test_squad_post_eval():
    tok = _build_fast_tokenizer()
    ds = bench.build_qa_rows(_examples(), tok, {"segment": False}, cap=None, max_len=32, seed=0)
    gold_s, gold_e = ds[0]["start_positions"], ds[0]["end_positions"]

    class OracleModel(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **k):
            b, n = input_ids.shape
            sl, el = torch.full((b, n), -9.0), torch.full((b, n), -9.0)
            sl[:, gold_s] = 9.0
            el[:, gold_e] = 9.0
            return {"start_logits": sl, "end_logits": el}

    m = bench.squad_post_eval(OracleModel(), tok, ds, device="cpu", batch_size=4)
    check("oracle reaches F1=100", abs(m["f1"] - 100.0) < 1e-6, f"f1={m['f1']}")
    check("oracle reaches EM=100", abs(m["exact_match"] - 100.0) < 1e-6, f"em={m['exact_match']}")


def test_neobert_head_forward():
    class StubEncoder(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **k):
            b, n = input_ids.shape
            return SimpleNamespace(last_hidden_state=torch.randn(b, n, 16))

    class StubBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16)
            self.model = StubEncoder()

    from salt3_common import NeoBERTQuestionAnswering
    qa = NeoBERTQuestionAnswering(StubBase())
    ids = torch.randint(0, 5, (2, 7))
    out = qa(input_ids=ids, attention_mask=torch.ones_like(ids),
             start_positions=torch.tensor([1, 2]), end_positions=torch.tensor([3, 4]))
    check("neobert head start_logits shape", tuple(out["start_logits"].shape) == (2, 7))
    check("neobert head loss finite", torch.isfinite(out["loss"]).item())


def test_skip_done_idempotency():
    specs = {"M": {"path": "x", "kind": "hf", "family": "F", "segment": False, "milestone_docs": None}}
    cfg = bench.MRCConfig(seeds=(42,))
    orig = bench.finetune_mrc
    calls = {"n": 0}

    def fake(spec, splits, *, seed, cfg, device="cuda"):
        calls["n"] += 1
        return {"f1": 50.0, "exact_match": 25.0, "n_params": 1000}

    bench.finetune_mrc = fake
    try:
        with tempfile.TemporaryDirectory() as d:
            jl = Path(d) / "r.jsonl"
            bench.run_benchmark(specs, {}, jl, cfg=cfg, device="cpu")
            bench.run_benchmark(specs, {}, jl, cfg=cfg, device="cpu")
            from salt3_common import read_jsonl
            rows = read_jsonl(jl)
        check("second run skipped (1 train call)", calls["n"] == 1, f"calls={calls['n']}")
        check("exactly one result row", len(rows) == 1, f"rows={len(rows)}")
        check("row carries family+segment", rows[0]["family"] == "F" and rows[0]["segment"] is False)
    finally:
        bench.finetune_mrc = orig


def test_milestone_kind_priority():
    """Auto-discovery prefers decay > cooled > stable per budget, records the chosen kind."""
    import salt3_mrc_models as models

    with tempfile.TemporaryDirectory() as d:
        arm = Path(d) / models.OUR_ARM
        for name in ("milestone_5000000_decay", "milestone_5000000_cooled", "milestone_5000000_stable",
                     "milestone_4000000_cooled", "milestone_4000000_stable", "milestone_3000000_stable"):
            (arm / name).mkdir(parents=True)
            (arm / name / "model.safetensors").write_bytes(b"x")
        got = {s["milestone_docs"]: s["checkpoint_kind"] for s in models.milestone_specs(d).values()}
        check("5M resolves to decay", got.get(5_000_000) == "decay", str(got))
        check("4M resolves to cooled", got.get(4_000_000) == "cooled", str(got))
        check("3M resolves to stable", got.get(3_000_000) == "stable", str(got))


def main():
    print("MRC benchmark CPU checks")
    for fn in (test_answerable_filter, test_span_alignment_unsegmented, test_segment_remap,
               test_slow_wordlevel_path, test_squad_post_eval, test_neobert_head_forward,
               test_skip_done_idempotency, test_milestone_kind_priority):
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
