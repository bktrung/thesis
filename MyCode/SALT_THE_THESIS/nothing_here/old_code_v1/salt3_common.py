from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_environment() -> None:
    os.environ.setdefault("HF_HUB_TRUST_REMOTE_CODE", "1")
    os.environ.setdefault("TRUST_REMOTE_CODE", "1")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def mount_drive_if_available(force_remount: bool = False) -> bool:
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=force_remount)
        return True
    except Exception:
        return False


def project_root(project_name: str = "SALT3") -> Path:
    if Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive") / project_name
    return Path.cwd() / project_name


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_ref(root: str | Path, ref: str | Path) -> Path:
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return Path(root) / ref_path


def run_dir(root: str | Path, run_name: str) -> Path:
    return Path(root) / "runs" / run_name


def init_model_ref(init_name: str) -> str:
    return f"init/{init_name}/model"


def final_model_ref(run_name: str) -> str:
    return f"runs/{run_name}/final_model"


def get_last_checkpoint(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    try:
        from transformers.trainer_utils import get_last_checkpoint as hf_get_last_checkpoint

        return hf_get_last_checkpoint(str(output_dir))
    except Exception:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1,
        )
        return str(checkpoints[-1]) if checkpoints else None


def _flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_config(value, full_key + "."))
        else:
            flat[full_key] = value
    return flat


def _warn_ignored_config_edits(passed: dict[str, Any] | None, loaded: dict[str, Any]) -> None:
    """Resume runs train with the saved on-disk config; notebook edits to the config
    cells are silently dead in that mode, so surface every value that differs."""
    if not passed:
        return
    skip = {"mode", "run_name", "base_model_ref", "created_at", "strategy"}
    passed_flat = _flatten_config(passed)
    loaded_flat = _flatten_config(loaded)
    diffs = [
        f"{key}: saved={loaded_flat[key]!r} (USED) vs notebook={value!r} (ignored)"
        for key, value in sorted(passed_flat.items())
        if key.split(".")[0] not in skip and key in loaded_flat and loaded_flat[key] != value
    ]
    diffs += [
        f"{key}: notebook={passed_flat[key]!r} not present in saved config (ignored)"
        for key in sorted(passed_flat)
        if key.split(".")[0] not in skip and key not in loaded_flat
    ]
    if diffs:
        print("⚠ MODE='resume' trains with the saved run_config.json — these notebook edits are IGNORED:")
        for diff in diffs:
            print(f"    {diff}")
        print("  Use MODE='new'/'continue' with a fresh RUN_NAME to train with the edited values.")


def create_or_load_run(
    root: str | Path,
    mode: str,
    run_name: str,
    base_model_ref: str | None,
    config: dict[str, Any],
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    root = ensure_dir(root)
    rd = run_dir(root, run_name)
    cfg_path = rd / "run_config.json"

    if mode not in {"new", "resume", "continue"}:
        raise ValueError("mode must be one of: new, resume, continue")

    if mode == "new" and rd.exists() and not allow_overwrite:
        raise FileExistsError(f"Run already exists: {rd}. Use MODE='resume' or choose a new RUN_NAME.")
    if mode == "resume" and not cfg_path.exists():
        raise FileNotFoundError(f"Cannot resume; missing run_config.json in {rd}")
    if mode == "continue" and (rd.exists() and not allow_overwrite):
        raise FileExistsError(f"Continuation run already exists: {rd}. Choose a new RUN_NAME or use MODE='resume'.")

    if mode == "resume":
        loaded = read_json(cfg_path)
        loaded["mode"] = mode
        _warn_ignored_config_edits(config, loaded)
        return add_run_paths(root, loaded)

    if not base_model_ref:
        raise ValueError("BASE_MODEL_REF is required for MODE='new' and MODE='continue'")

    rd.mkdir(parents=True, exist_ok=True)
    run_config = dict(config)
    run_config.update(
        {
            "mode": mode,
            "run_name": run_name,
            "base_model_ref": base_model_ref,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_json(cfg_path, run_config)
    return add_run_paths(root, run_config)


def add_run_paths(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    rd = run_dir(root, config["run_name"])
    config = dict(config)
    config["project_root"] = str(root)
    config["run_dir"] = str(rd)
    config["checkpoint_dir"] = str(rd / "checkpoints")
    config["final_model_dir"] = str(rd / "final_model")
    config["metrics_path"] = str(rd / "metrics.jsonl")
    config["plots_dir"] = str(rd / "plots")
    for key in ("checkpoint_dir", "plots_dir"):
        ensure_dir(config[key])
    return config


def validate_model_artifact(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    required_any = [path / "model.safetensors", path / "pytorch_model.bin"]
    if not any(p.exists() for p in required_any):
        raise FileNotFoundError(f"No model weights found in {path}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {path}")
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "sentencepiece.bpe.model"]
    if not any((path / name).exists() for name in tokenizer_files):
        raise FileNotFoundError(f"No tokenizer files found in {path}")


def load_model_safe(
    name_or_path: str | Path,
    model_cls=None,
    device: str | torch.device = "cpu",
    allow_partial: bool = False,
    **config_overrides,
):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForMaskedLM

    name_or_path = str(name_or_path)
    if model_cls is None:
        model_cls = AutoModelForMaskedLM
    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    for k, v in config_overrides.items():
        setattr(config, k, v)
    model = model_cls.from_config(config, trust_remote_code=True)

    if os.path.isdir(name_or_path):
        safe_path = os.path.join(name_or_path, "model.safetensors")
        bin_path = os.path.join(name_or_path, "pytorch_model.bin")
        if os.path.exists(safe_path):
            state_dict = load_file(safe_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights in {name_or_path}")
    else:
        try:
            safe_path = hf_hub_download(repo_id=name_or_path, filename="model.safetensors")
            state_dict = load_file(safe_path)
        except Exception:
            bin_path = hf_hub_download(repo_id=name_or_path, filename="pytorch_model.bin")
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  ⚠ load_model_safe: {len(result.missing_keys)} MISSING keys (left at random init):")
        for k in result.missing_keys[:20]:
            print(f"      {k}")
        if len(result.missing_keys) > 20:
            print(f"      ... and {len(result.missing_keys) - 20} more")
        ffn_missing = [k for k in result.missing_keys if "ffn" in k or "w12" in k or "w3" in k]
        if ffn_missing:
            print(f"  ⚠⚠ CRITICAL: {len(ffn_missing)} FFN/SwiGLU weights missing!")
            print(f"     This means >50% of encoder params are random: the SwiGLU module in the")
            print(f"     loaded model.py no longer exposes the w12/w3 parameter names the saved")
            print(f"     weights use. Check model.py's SwiGLU (xformers or the patched fallback).")
            if not allow_partial:
                raise RuntimeError(
                    f"{len(ffn_missing)} FFN/SwiGLU weight tensors missing when loading {name_or_path}; "
                    "refusing to return a half-random encoder (training it would waste the whole run). "
                    "Pass allow_partial=True to override."
                )
    if result.unexpected_keys:
        print(f"  ⚠ load_model_safe: {len(result.unexpected_keys)} unexpected keys (not loaded):")
        for k in result.unexpected_keys[:10]:
            print(f"      {k}")
        if len(result.unexpected_keys) > 10:
            print(f"      ... and {len(result.unexpected_keys) - 10} more")
    if not result.missing_keys and not result.unexpected_keys:
        print(f"  ✓ load_model_safe: all {len(state_dict)} keys loaded successfully")
    return model.to(device)


def _patch_model_py_xformers(path: Path) -> None:
    """Replace xformers.ops.SwiGLU import with a compatible pure-PyTorch fallback.

    The HuggingFace NeoBERT model.py hard-imports xformers just for SwiGLU.
    This patch removes that dependency while keeping identical parameter names
    (w12, w3) so load_state_dict matches the saved weights exactly.
    """
    text = path.read_text(encoding="utf-8")
    if "class SwiGLU(nn.Module)" in text:
        return  # already patched; re-patching would nest the try/except wrapper
    if "from xformers.ops import SwiGLU" not in text:
        # Silent no-op here would ship an unpatched model.py that hard-requires
        # xformers in whatever environment loads the artifact next.
        print(f"  ⚠ {path}: xformers SwiGLU import not found — upstream model.py changed; fallback patch NOT applied.")
        return
    swiglu_impl = (
        "try:\n"
        "    from xformers.ops import SwiGLU\n"
        "except ImportError:\n"
        "    class SwiGLU(nn.Module):\n"
        "        def __init__(self, in_features, hidden_features, out_features=None, bias=True, **_kw):\n"
        "            super().__init__()\n"
        "            out_features = out_features or in_features\n"
        "            self.w12 = nn.Linear(in_features, hidden_features * 2, bias=bias)\n"
        "            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)\n"
        "        def forward(self, x):\n"
        "            x12 = self.w12(x)\n"
        "            x1, x2 = x12.chunk(2, dim=-1)\n"
        "            hidden = torch.nn.functional.silu(x1) * x2\n"
        "            return self.w3(hidden)\n"
    )
    text = text.replace("from xformers.ops import SwiGLU", swiglu_impl)
    path.write_text(text, encoding="utf-8")


def copy_neobert_remote_files(model_dir: str | Path, repo_id: str = "chandar-lab/NeoBERT") -> None:
    from huggingface_hub import hf_hub_download

    model_dir = ensure_dir(model_dir)
    for filename in ("model.py", "rotary.py"):
        dst = model_dir / filename
        shutil.copy(hf_hub_download(repo_id=repo_id, filename=filename), dst)
    _patch_model_py_xformers(model_dir / "model.py")


def extract_embedding_weight(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "encoder") and isinstance(model.encoder, nn.Embedding):
        return model.encoder.weight.detach()
    if hasattr(model, "model") and hasattr(model.model, "encoder") and isinstance(model.model.encoder, nn.Embedding):
        return model.model.encoder.weight.detach()
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb.weight.detach()
    except Exception:
        pass
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
        return model.embeddings.word_embeddings.weight.detach()
    for _, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return module.weight.detach()
    raise ValueError(f"Cannot find embedding matrix in {model.__class__.__name__}")


def fit_embedding_to_decoder_map(source_embeddings: torch.Tensor, source_decoder_weight: torch.Tensor):
    """Fit one global linear map M with source_embeddings @ M ≈ source_decoder_weight.

    The output LM head and the input embeddings are different learned spaces, but
    across the whole vocabulary (vocab >> hidden) the relationship is well captured
    by a single overdetermined least-squares fit — far more stable than the per-token
    k≤64 lstsq the SALT paper uses for embeddings (the paper only ever transferred a
    tied head, so it never needed a decoder map for an untied model like NeoBERT).

    Applying M to the SALT-projected embeddings yields decoder rows that are
    consistent with those embeddings, which beats per-token decoder projection
    empirically (~10.7 vs ~15.0 step-0 MLM loss on Vietnamese).

    Returns (M[hidden, hidden], relative_residual). High residual → no global linear
    head exists and tying would be the fallback.
    """
    E = source_embeddings.float()
    D = source_decoder_weight.float()
    M = torch.linalg.lstsq(E, D).solution
    resid = float((E @ M - D).norm() / D.norm().clamp(min=1e-8))
    return M, resid


def target_unigram_logfreq_bias(
    tokenizer,
    vocab_size: int,
    num_docs: int = 20000,
    dataset_id: str = "uonlp/CulturaX",
    lang: str = "vi",
    smoothing: float = 1.0,
):
    """Stream target-language text and return a log-frequency decoder bias.

    NeoBERT's trained decoder bias is a large frequency prior (norm ≈ 229); zeroing
    it discards the single strongest MLM signal. This rebuilds that prior in the
    *target* language. Counting uses the SAME (pruned) tokenizer that training uses,
    so ids align with the embedding/decoder rows.

    Returns (logfreq_bias[vocab_size], counts[vocab_size]).
    """
    from datasets import load_dataset

    counts = torch.zeros(vocab_size, dtype=torch.float64)
    stream = load_dataset(dataset_id, lang, split="train", streaming=True).take(num_docs)
    for batch in stream.iter(batch_size=1000):
        encoded = tokenizer(
            batch["text"],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for row in encoded["input_ids"]:
            if row:
                idx = torch.tensor(row, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < vocab_size)]
                counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
    probs = (counts + smoothing) / (counts.sum() + smoothing * vocab_size)
    return probs.log().float(), counts


def embedding_fingerprint(model: nn.Module, sample_rows: int = 256) -> dict[str, Any]:
    emb = extract_embedding_weight(model).float().cpu()
    rows = emb[: min(sample_rows, emb.shape[0])].contiguous().numpy().tobytes()
    return {
        "shape": list(emb.shape),
        "mean_norm": float(emb.norm(dim=1).mean()),
        "std_norm": float(emb.norm(dim=1).std()),
        "sha256_first_rows": hashlib.sha256(rows).hexdigest(),
    }


def tokenizer_fingerprint(tokenizer, sample_texts: list[str] | None = None) -> dict[str, Any]:
    if sample_texts is None:
        sample_texts = [
            "Xin chào Việt Nam",
            "Hôm nay thời tiết rất đẹp.",
            "Mô hình ngôn ngữ học tiếng Việt.",
        ]
    vocab = tokenizer.get_vocab()
    payload = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "vocab_items": sorted((str(token), int(idx)) for token, idx in vocab.items()),
        "special_ids": {
            "pad": tokenizer.pad_token_id,
            "unk": tokenizer.unk_token_id,
            "mask": tokenizer.mask_token_id,
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "cls": tokenizer.cls_token_id,
            "sep": tokenizer.sep_token_id,
        },
        "sample_ids": [tokenizer.encode(text, add_special_tokens=False) for text in sample_texts],
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "sha256": digest,
        "short": digest[:12],
        "tokenizer_class": payload["tokenizer_class"],
        "vocab_size": payload["vocab_size"],
        "special_ids": payload["special_ids"],
        "sample_ids": payload["sample_ids"],
    }


def inspect_tokenized_dataset(dataset, tokenizer, name: str = "dataset", n_examples: int = 2, n_tokens: int = 160) -> None:
    print(f"── Tokenized {name} sanity ──")
    print(f"  Rows: {len(dataset):,}")
    vocab_size = len(tokenizer)
    for i in range(min(n_examples, len(dataset))):
        ids = list(dataset[i]["input_ids"])
        max_id = max(ids) if ids else -1
        min_id = min(ids) if ids else -1
        oob = sum(1 for token_id in ids if token_id < 0 or token_id >= vocab_size)
        unk = sum(1 for token_id in ids if token_id == tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else 0
        decoded = tokenizer.decode(ids[:n_tokens], skip_special_tokens=False)
        compact = " ".join(decoded.replace("\n", " ").split())
        print(f"  Example {i}: len={len(ids)} min_id={min_id} max_id={max_id} oob={oob} unk={unk}")
        print(f"    decode: {compact[:500]}")


class JsonlMetricsCallback:
    def __init__(self, metrics_path: str | Path):
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self, outer: "JsonlMetricsCallback"):
                self.outer = outer

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return control
                record = {
                    "step": int(state.global_step),
                    "epoch": None if state.epoch is None else float(state.epoch),
                    "timestamp": time.time(),
                    "session": self.outer.session,
                }
                record.update({k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool)) or v is None})
                append_jsonl(self.outer.metrics_path, record)
                return control

        self.metrics_path = Path(metrics_path)
        # Distinguishes trajectories when a crashed run restarts and re-logs steps
        # that an abandoned earlier session already wrote to the same file.
        self.session = time.strftime("%Y%m%d-%H%M%S")
        self.callback = _Callback(self)


def plot_training_curves(metrics_path: str | Path, save_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    rows = read_jsonl(metrics_path)

    def latest_by_step(pairs):
        # A restarted run re-logs steps an abandoned session already wrote; rows are
        # appended chronologically, so keeping the last record per step plots the
        # live trajectory instead of a curve that doubles back on itself.
        by_step: dict[int, float] = {}
        for step, value in pairs:
            by_step[step] = value
        return sorted(by_step.items())

    train = latest_by_step((r["step"], r["loss"]) for r in rows if "loss" in r)
    evals = latest_by_step((r["step"], r["eval_loss"]) for r in rows if "eval_loss" in r)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if train:
        axes[0].plot([x for x, _ in train], [y for _, y in train])
    if evals:
        axes[1].plot([x for x, _ in evals], [y for _, y in evals], marker="o")
    axes[0].set_title("Training loss")
    axes[1].set_title("Eval loss")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def make_neo_mlm_trainer_class():
    from transformers import Trainer

    class NeoMLMTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # compute_loss below returns a mean CE and never reads num_items_in_batch.
            # NeoBERT's forward has **kwargs, which makes transformers >=4.49 set
            # model_accepts_loss_kwargs=True and SKIP dividing the accumulated loss by
            # gradient_accumulation_steps — silently scaling gradients (and effective
            # LR) by the accumulation factor. Force the flag off so the Trainer keeps
            # normalizing; harmless on older transformers.
            self.model_accepts_loss_kwargs = False

        def _neo_forward_with_loss(self, model, inputs):
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            labels = inputs.get("labels")
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            return loss, logits

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss, logits = self._neo_forward_with_loss(model, inputs)
            return (loss, {"logits": logits}) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            model.eval()
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, logits = self._neo_forward_with_loss(model, inputs)
            if prediction_loss_only:
                return loss, None, None
            return loss, logits, None

    return NeoMLMTrainer


class NeoBERTSequenceClassifier(nn.Module):
    """Matches official NeoBERTForSequenceClassification: [CLS] -> dense -> tanh -> classifier."""

    def __init__(self, base_model: nn.Module, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(base_model.config, "d_model")
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = base_model.config
        self.config.num_labels = num_labels

    def _get_encoder(self):
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        return self.base_model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        output = self._get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        x = output.last_hidden_state[:, 0]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.classifier.out_features == 1:
                # regression head (STS): MSE on the single output column
                loss = F.mse_loss(logits.view(-1), labels.view(-1).float())
            else:
                loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


class NeoBERTTokenClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(base_model.config, "d_model")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = base_model.config
        self.config.num_labels = num_labels

    def _get_encoder(self):
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        return self.base_model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        output = self._get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(output.last_hidden_state))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}


class NeoBERTQuestionAnswering(nn.Module):
    def __init__(self, base_model: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(base_model.config, "d_model")
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.config = base_model.config

    def _get_encoder(self):
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        return self.base_model

    def forward(self, input_ids=None, attention_mask=None, start_positions=None, end_positions=None, **kwargs):
        output = self._get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.qa_outputs(self.dropout(output.last_hidden_state))
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss = (
                F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
                + F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            ) / 2
        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}


def make_mlm_collator(tokenizer, mlm_probability: float = 0.20):
    from transformers import DataCollatorForLanguageModeling

    class FullMaskMLMCollator(DataCollatorForLanguageModeling):
        """NeoBERT was pretrained with 100% [MASK] replacement (no random/keep-original).
        The standard 80/10/10 split causes confidently-wrong predictions at non-MASK
        positions because the encoder was never trained to see real tokens there."""

        def torch_mask_tokens(self, inputs, special_tokens_mask=None, **kwargs):
            labels = inputs.clone()
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask.bool()
                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100
            inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            return inputs, labels

    return FullMaskMLMCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=8,
    )


def _slice_chunks(train_ds, eval_ds, num_chunks: int | None, eval_ratio: float):
    """Take a smaller experiment slice from a larger cached train/eval pair."""
    if num_chunks is not None and num_chunks < len(train_ds) + len(eval_ds):
        n_eval = max(1, int(num_chunks * eval_ratio))
        n_train = num_chunks - n_eval
        if n_train < len(train_ds):
            train_ds = train_ds.select(range(n_train))
        if n_eval < len(eval_ds):
            eval_ds = eval_ds.select(range(n_eval))
    print(f"  Using {len(train_ds):,} train + {len(eval_ds):,} eval chunks")
    return train_ds, eval_ds


def make_mlm_datasets(
    tokenizer,
    cache_dir: str | Path,
    num_examples: int,
    max_seq_len: int,
    num_chunks: int | None = None,
    eval_ratio: float = 0.02,
    seed: int = 42,
):
    """Tokenize CulturaX-vi, cache the full result to Drive, then return a subset.

    Design:
        - Each *document* is routed to train or validation before packing, so no
            document contributes tokens to both splits. A chunk-level random split
            would put adjacent 1024-token chunks of the same document on both sides,
            leaking near-duplicate text into eval and biasing early stopping,
            best-model selection, and the reported perplexity.
        - The cache key is ``(num_examples, max_seq_len, eval_ratio, seed, tokenizer
            fingerprint)`` so streaming + tokenization only happens once for a given
            document count and exact tokenizer.
        - After caching, ``num_chunks`` controls how many chunks are actually used.
            Pass ``None`` to use the entire cache.
        - A 400K-chunk experiment and a 3M-chunk experiment can share the same 3M cache
            if they use the same tokenizer, set ``num_examples=3_000_000`` for both,
            and vary ``num_chunks``.

    Typical sizes on disk (Arrow format, memory-mapped, loads in seconds):
      - 400K docs @ 1024 → ~400K chunks → ~1.6 GB
      - 3M docs   @ 1024 → ~3M chunks   → ~12 GB

    Args:
        tokenizer: HF tokenizer (must already be the pruned SALT tokenizer).
        cache_dir: Root directory for dataset caches (usually ``PROJECT_ROOT / 'datasets'``).
        num_examples: Number of CulturaX documents to stream and tokenize.
        max_seq_len: Chunk length in tokens.
        num_chunks: Optional cap on how many chunks to use from the cache.
            ``None`` means use all chunks. This lets you build one large cache
            and slice smaller training sets from it without re-tokenizing.
        eval_ratio: Fraction of documents to hold out for validation.
        seed: Random seed for the document-level train/eval routing.
    """
    from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset, load_from_disk

    tok_fp = tokenizer_fingerprint(tokenizer)
    # 'docsplit' distinguishes these caches from older ones built with the leaky
    # chunk-level random split, so a stale cache is never silently reused.
    cache_key = f"culturax_vi_{num_examples}_seq{max_seq_len}_tok{tok_fp['short']}_ev{eval_ratio:g}_s{seed}_docsplit"
    cache_dir = Path(cache_dir) / cache_key
    print(f"Tokenizer fingerprint: {tok_fp['short']} ({tok_fp['tokenizer_class']}, vocab={tok_fp['vocab_size']})")
    print(f"Dataset cache key    : {cache_dir.name}")

    if cache_dir.exists():
        print(f"Loading cached dataset from {cache_dir} ...")
        loaded = load_from_disk(str(cache_dir))
        return _slice_chunks(loaded["train"], loaded["validation"], num_chunks, eval_ratio)

    print(f"Streaming {num_examples:,} CulturaX-vi docs (this is the slow part, ~1 hr for 3M)...")
    print(f"  Result will be cached to {cache_dir} for instant reload next time.")
    raw_ds = load_dataset("uonlp/CulturaX", "vi", split="train", streaming=True).take(num_examples)
    features = Features({"input_ids": Sequence(Value("int32"), length=max_seq_len), "split": Value("string")})
    generator_cache_dir = cache_dir.parent / ".hf_tmp"
    generator_cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_chunks():
        # Route whole documents to train or validation, packing each split's token
        # stream separately, so no document's text appears on both sides of the split.
        rng = random.Random(seed)
        carry = {"train": [], "validation": []}
        doc_count = 0
        chunk_count = 0
        for batch in raw_ds.iter(batch_size=1000):
            encoded = tokenizer(
                batch["text"],
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            for row in encoded["input_ids"]:
                split_name = "validation" if rng.random() < eval_ratio else "train"
                carry[split_name].extend(row)

            for split_name, buf in carry.items():
                usable = (len(buf) // max_seq_len) * max_seq_len
                for start in range(0, usable, max_seq_len):
                    yield {"input_ids": buf[start : start + max_seq_len], "split": split_name}
                    chunk_count += 1
                carry[split_name] = buf[usable:]

            doc_count += len(batch["text"])
            if doc_count % 50_000 < 1000:
                print(f"  {doc_count:>10,} docs -> {chunk_count:>10,} chunks")

        print(f"  Total: {doc_count:,} docs -> {chunk_count:,} chunks of {max_seq_len} tokens")

    # Build the Arrow cache incrementally so 3M-doc runs do not retain the whole corpus in RAM.
    full = Dataset.from_generator(
        generate_chunks,
        features=features,
        cache_dir=str(generator_cache_dir),
        fingerprint=f"{cache_key}_streamed_v4",
    )
    wrapped = DatasetDict(
        {
            name: full.filter(
                lambda splits, _name=name: [s == _name for s in splits],
                input_columns=["split"],
                batched=True,
            ).remove_columns("split")
            for name in ("train", "validation")
        }
    )
    print(f"  Saving cache to {cache_dir} ...")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    wrapped.save_to_disk(str(cache_dir))
    print(f"  Cache saved ({len(wrapped['train']):,} train + {len(wrapped['validation']):,} eval)")
    return _slice_chunks(wrapped["train"], wrapped["validation"], num_chunks, eval_ratio)


def training_args(output_dir: str | Path, **overrides):
    from transformers import TrainingArguments

    defaults = dict(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=-1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        tf32=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def train_with_run_resume(trainer, mode: str):
    checkpoint = get_last_checkpoint(trainer.args.output_dir) if mode == "resume" else None
    if checkpoint:
        print(f"Resuming trainer checkpoint: {checkpoint}")
    elif mode == "resume":
        print("Resume mode selected, but no trainer checkpoint exists yet; starting this run from its configured base model.")
    return trainer.train(resume_from_checkpoint=checkpoint)


def eval_with_perplexity(trainer, dataset=None, metric_key_prefix: str = "eval") -> dict[str, Any]:
    """Run evaluation and add perplexity (capped at exp(20) to avoid overflow)."""
    import math

    metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=metric_key_prefix)
    loss_key = f"{metric_key_prefix}_loss"
    if loss_key in metrics:
        ppl = math.exp(min(metrics[loss_key], 20))
        metrics[f"{metric_key_prefix}_perplexity"] = ppl
    return metrics


def diagnose_model_health(model, tokenizer, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Cheap diagnostic to verify a SALT-initialized NeoBERT loaded correctly.

    Checks:
      1. Embedding norms (trained SALT ≈ 0.3–0.5; random init ≈ 0.01–0.03)
      2. Transformer layer weight stats (trained vs random)
      3. Forward pass: logits entropy on sample text
         - High entropy (uniform) = random decoder, expected before CPT
         - Low entropy + wrong predictions = broken transformer
         - Reasonable predictions = model is healthy
      4. Expected random-model loss = log(vocab_size) for comparison

    Returns a dict with diagnostic values for programmatic checks.
    """
    import math

    info: dict[str, Any] = {}
    vocab_size = model.config.vocab_size
    info["vocab_size"] = vocab_size
    info["expected_random_loss"] = math.log(vocab_size)

    # 1. Embedding norms
    emb = extract_embedding_weight(model).float().cpu()
    norms = emb.norm(dim=1)
    info["emb_mean_norm"] = float(norms.mean())
    info["emb_std_norm"] = float(norms.std())

    print(f"── Model health diagnostic ──")
    print(f"  Vocab size              : {vocab_size}")
    print(f"  Expected random loss    : {info['expected_random_loss']:.2f}  (= log({vocab_size}))")
    print(f"  Embedding mean norm     : {info['emb_mean_norm']:.4f}")
    print(f"  Embedding std norm      : {info['emb_std_norm']:.4f}")

    # 2. Transformer weight stats — check if encoder layers are trained or random
    encoder_params = []
    encoder_random_count = 0
    encoder_total_count = 0
    for name, p in model.named_parameters():
        is_layer_param = (
            "transformer_encoder" in name
            or "encoder.layer" in name
            or "model.layers." in name
        )
        if is_layer_param:
            std = p.detach().float().std().item()
            encoder_params.append((name, std))
            encoder_total_count += 1
            # Untrained tensors: normal init (std ~0.02), NeoBERT's actual
            # uniform_(-0.02, 0.02) init (std = 0.02/sqrt(3) ~0.0115), or
            # constant-filled tensors like fresh norm weights (std ~0).
            uniform_init_std = 0.02 / math.sqrt(3)
            if abs(std - 0.02) < 0.005 or abs(std - uniform_init_std) < 0.002 or std < 0.005:
                encoder_random_count += 1

    if encoder_params:
        stds = [s for _, s in encoder_params]
        mean_std = sum(stds) / len(stds)
        info["encoder_mean_weight_std"] = mean_std
        info["encoder_random_looking"] = encoder_random_count
        info["encoder_total_params"] = encoder_total_count
        print(f"  Encoder mean weight std : {mean_std:.6f}")
        print(f"  Encoder params          : {encoder_total_count} tensors, "
              f"{encoder_random_count} look random-init")
        if encoder_random_count > encoder_total_count * 0.5:
            print(f"  ⚠ MOST ENCODER WEIGHTS LOOK RANDOM — load_state_dict likely failed!")
            print(f"    This is the #1 cause of eval_loss >> log(vocab).")
            info["diagnosis"] = "BROKEN — encoder weights not loaded"
        else:
            print(f"  ✓ Encoder weights appear trained")
    else:
        print(f"  ⚠ No encoder parameters found — check model structure")
        info["diagnosis"] = "UNKNOWN — no encoder params found"

    # 3. Forward pass sanity
    model.eval()
    test_texts = ["Xin chào Việt Nam", "Hôm nay thời tiết rất đẹp"]
    for text in test_texts:
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        has_nan = torch.isnan(logits).any().item()
        probs = torch.softmax(logits[0], dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1).mean().item()
        max_entropy = math.log(vocab_size)
        pred_ids = logits[0].argmax(dim=-1)
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids.cpu().tolist())
        print(f"  '{text}':")
        print(f"    NaN: {'YES ⚠' if has_nan else 'no'} | "
              f"entropy: {entropy:.2f}/{max_entropy:.2f} ({entropy/max_entropy:.0%} of max)")
        print(f"    top predictions: {' '.join(pred_tokens[:8])}")

    # 4. Mask-position probe — much closer to the actual MLM objective than argmax on plain text.
    if getattr(tokenizer, "mask_token", None) is not None and getattr(tokenizer, "mask_token_id", None) is not None:
        mask_text = f"Xin chào {tokenizer.mask_token} Nam"
        enc = tokenizer(mask_text, return_tensors="pt").to(device)
        mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False).squeeze(-1)
        if len(mask_pos) == 1:
            with torch.no_grad():
                out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out["logits"]
            mask_logits = logits[0, mask_pos.item()]
            mask_probs = torch.softmax(mask_logits, dim=-1)
            mask_entropy = -(mask_probs * mask_probs.log().clamp(min=-100)).sum().item()
            top_ids = mask_logits.topk(8).indices.cpu().tolist()
            top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
            info["mask_probe_entropy"] = float(mask_entropy)
            info["mask_probe_top_tokens"] = top_tokens
            print(f"  Mask probe '{mask_text}':")
            print(f"    entropy: {mask_entropy:.2f}/{max_entropy:.2f} ({mask_entropy/max_entropy:.0%} of max)")
            print(f"    top predictions: {' '.join(top_tokens)}")

    if "diagnosis" not in info:
        info["diagnosis"] = "OK — encoder weights loaded, forward pass clean"

    print(f"\n  Diagnosis: {info['diagnosis']}")
    return info


# ══════════════════════════════════════════════════════════════════════════════
# Downstream evaluation helpers (shared by notebook 03)
# One protocol for every model/task: regression head, early stopping, hybrid HP
# grid + multi-seed, Vietnamese word segmentation, and statistical reporting.
# ══════════════════════════════════════════════════════════════════════════════

def compute_regression_metrics(eval_pred) -> dict[str, float]:
    """STS metrics: Spearman (primary), Pearson, MSE. Logits are a single column."""
    from scipy.stats import pearsonr, spearmanr

    logits, labels = eval_pred
    preds = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {
        "spearman": float(spearmanr(preds, labels).correlation),
        "pearson": float(pearsonr(preds, labels)[0]),
        "mse": float(np.mean((preds - labels) ** 2)),
    }


def spearman_pearson(preds, labels) -> tuple[float, float]:
    from scipy.stats import pearsonr, spearmanr

    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return float(spearmanr(preds, labels).correlation), float(pearsonr(preds, labels)[0])


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI over a small sample (e.g. per-seed metric values)."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_pvalue(labels, preds_a, preds_b, metric_fn, n_boot: int = 10000, seed: int = 0):
    """Two-sided paired bootstrap on metric(a) - metric(b) over the SAME test set.

    metric_fn(labels, preds) -> float (e.g. accuracy, macro-F1, Spearman).
    Returns (observed_diff, p_two_sided, (ci_lo, ci_hi)). a is the model under test.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = labels.shape[0]
    obs = metric_fn(labels, preds_a) - metric_fn(labels, preds_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(labels[idx], preds_a[idx]) - metric_fn(labels[idx], preds_b[idx])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(min(p, 1.0)), (float(lo), float(hi))


def make_early_stopping(patience: int = 3):
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=patience)


# ── Vietnamese word segmentation (VnCoreNLP) for PhoBERT / ViDeBERTa baselines ──
_VNCORENLP = {"model": None}


def get_vncorenlp(save_dir: str | Path = "/content/vncorenlp"):
    """Lazy-load a cached VnCoreNLP word segmenter. Returns None if unavailable.

    Failures are retried on the next call rather than latched: a transient blip
    (download hiccup, JVM init race) must not permanently disable segmentation,
    or the segmented baselines would silently fine-tune on unsegmented text.
    """
    if _VNCORENLP["model"] is not None:
        return _VNCORENLP["model"]
    try:
        import py_vncorenlp

        save_dir = str(ensure_dir(save_dir))
        try:
            py_vncorenlp.download_model(save_dir=save_dir)
        except Exception:
            pass  # already downloaded
        _VNCORENLP["model"] = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
    except Exception as exc:
        print(f"  ⚠ VnCoreNLP unavailable ({exc})")
        return None
    return _VNCORENLP["model"]


def segment_texts(texts, save_dir: str | Path = "/content/vncorenlp"):
    """Word-segment a list of strings (e.g. 'Hà Nội' -> 'Hà_Nội'). No-op if unavailable."""
    seg = get_vncorenlp(save_dir)
    if seg is None:
        return list(texts)
    out = []
    for t in texts:
        if not t:
            out.append(t)
            continue
        try:
            out.append(" ".join(seg.word_segment(t)))
        except Exception:
            out.append(t)
    return out


def maybe_segment(texts, model_spec: dict, save_dir: str | Path = "/content/vncorenlp"):
    """Segment iff the model spec opts in (spec['segment'] is True).

    Raises when a segment-requiring model cannot get a segmenter: fine-tuning
    PhoBERT/ViDeBERTa on unsegmented text degrades them by several points, which
    would silently invalidate the baseline comparison.
    """
    if not model_spec.get("segment"):
        return list(texts)
    if get_vncorenlp(save_dir) is None:
        raise RuntimeError(
            "VnCoreNLP segmenter unavailable but this model requires word-segmented input. "
            "Install Java + py_vncorenlp (see notebook setup cell) and rerun."
        )
    return segment_texts(texts, save_dir)
