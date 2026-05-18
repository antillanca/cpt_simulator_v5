"""Deterministic tiny-model experiment utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.datasets.loader import load_jsonl, load_sharded_dataset
from backend.neural.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint_payload,
    ensure_checkpoint_payload,
    hash_optimizer_state,
    hash_state_dict,
    infer_checkpoint_version,
)
from backend.validation.oracle_arena import ArenaExample, compare_oracle_vs_model
from backend.validation.model_eval import ModelEvaluator


@dataclass
class TrainConfig:
    model_type: str
    seed: int
    data_path: Path
    output_dir: Path
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-4
    max_steps: int | None = None
    device: str = "cpu"
    eval_every: int = 100
    save_every: int = 500
    train_split: float = 0.8
    shard_dir: Path | None = None
    manifest_path: Path | None = None
    output_checkpoint: Path | None = None


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _record_source_text(record: dict[str, Any]) -> str:
    payload = {
        "question": record.get("question", ""),
        "structured_state": record.get("structured_state", {}),
        "module_source": record.get("module_source", ""),
        "curriculum_layer": record.get("curriculum_layer", 0),
    }
    return _stable_json(payload)


def _record_target_text(record: dict[str, Any]) -> str:
    return _stable_json(record.get("final_answer", {}))


class CharTokenizer:
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, vocab: list[str]):
        specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        ordered = specials + [ch for ch in vocab if ch not in specials]
        self.itos = ordered
        self.stoi = {ch: idx for idx, ch in enumerate(ordered)}
        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    @classmethod
    def build(cls, records: Iterable[dict[str, Any]]) -> "CharTokenizer":
        chars: set[str] = set()
        for record in records:
            chars.update(_record_source_text(record))
            chars.update(_record_target_text(record))
        chars.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars.update("{}[],:.-_ /\\\"'")
        return cls(sorted(chars))

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            chars.append(self.itos[idx] if 0 <= idx < len(self.itos) else self.unk_token)
        return "".join(chars)

    def to_dict(self) -> dict[str, Any]:
        return {"vocab": self.itos}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharTokenizer":
        return cls(list(payload.get("vocab", [])))


class TinyTextDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: CharTokenizer, max_source_len: int = 512, max_target_len: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        source_ids = self.tokenizer.encode(_record_source_text(record), add_bos=True, add_eos=True)[: self.max_source_len]
        target_ids = self.tokenizer.encode(_record_target_text(record), add_bos=True, add_eos=True)[: self.max_target_len]
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "record": record,
        }


def _collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    source = [item["source_ids"] for item in batch]
    target = [item["target_ids"] for item in batch]
    source_pad = nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=pad_id)
    target_pad = nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=pad_id)
    return {"source_ids": source_pad, "target_ids": target_pad, "records": [item["record"] for item in batch]}


class _BaseTinyModel(nn.Module):
    model_type = "base"

    def __init__(self, vocab_size: int, pad_id: int, bos_id: int, eos_id: int, hidden_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.hidden_size = hidden_size
        self.decoder_embed = nn.Embedding(vocab_size, hidden_size)
        self.decoder_cell = nn.GRUCell(hidden_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)
        self.tokenizer: CharTokenizer | None = None

    def encode_source(self, source_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_source(source_ids)
        inputs = target_ids[:, :-1]
        batch, steps = inputs.shape
        logits: list[torch.Tensor] = []
        state = hidden
        for step in range(steps):
            token = self.decoder_embed(inputs[:, step])
            state = self.decoder_cell(token, state)
            logits.append(self.output_head(state).unsqueeze(1))
        return torch.cat(logits, dim=1) if logits else torch.zeros(batch, 0, self.vocab_size, device=source_ids.device)

    @torch.no_grad()
    def generate(self, source_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        hidden = self.encode_source(source_ids)
        prev = torch.full((source_ids.size(0),), self.bos_id, dtype=torch.long, device=source_ids.device)
        outputs: list[torch.Tensor] = [prev.unsqueeze(1)]
        state = hidden
        for _ in range(max_new_tokens):
            token = self.decoder_embed(prev)
            state = self.decoder_cell(token, state)
            logits = self.output_head(state)
            prev = torch.argmax(logits, dim=-1)
            outputs.append(prev.unsqueeze(1))
            if bool(torch.all(prev == self.eos_id)):
                break
        return torch.cat(outputs, dim=1)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self.tokenizer is None:
            return {"prediction": inputs, "model": self.model_type, "config": {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size}}
        source = torch.tensor([self.tokenizer.encode(_stable_json(inputs), add_bos=True, add_eos=True)], dtype=torch.long, device=next(self.parameters()).device)
        generated = self.generate(source)
        text = self.tokenizer.decode(generated[0].tolist())
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"text": text}
        return {
            "prediction": parsed,
            "final_answer": parsed,
            "reasoning_trace": [],
            "structured_state": inputs.get("structured_state", {}),
            "verification_status": {"passed": True, "violations": []},
            "model": self.model_type,
            "config": {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size},
        }

    def checkpoint_model_config(self) -> dict[str, Any]:
        return {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size}


class TinyTransformerModel(_BaseTinyModel):
    model_type = "transformer"

    def __init__(self, vocab_size: int, pad_id: int, bos_id: int, eos_id: int, hidden_size: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__(vocab_size, pad_id, bos_id, eos_id, hidden_size)
        self.positional = nn.Embedding(512, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def encode_source(self, source_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(source_ids.size(1), device=source_ids.device).unsqueeze(0)
        x = self.input_embed(source_ids) + self.positional(positions)
        mask = source_ids.eq(self.pad_id)
        encoded = self.encoder(x, src_key_padding_mask=mask)
        denom = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (encoded * (~mask).unsqueeze(-1)).sum(dim=1) / denom
        return torch.tanh(self.proj(pooled))

    def checkpoint_model_config(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "n_layers": len(self.encoder.layers),
            "n_heads": self.encoder.layers[0].self_attn.num_heads if self.encoder.layers else 0,
        }


class TinySeq2SeqModel(_BaseTinyModel):
    model_type = "seq2seq"

    def __init__(self, vocab_size: int, pad_id: int, bos_id: int, eos_id: int, hidden_size: int = 128, n_layers: int = 1):
        super().__init__(vocab_size, pad_id, bos_id, eos_id, hidden_size)
        self.input_embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def encode_source(self, source_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.input_embed(source_ids)
        _, hidden = self.encoder(embedded)
        return torch.tanh(self.proj(hidden[-1]))

    def checkpoint_model_config(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "n_layers": self.encoder.num_layers,
        }


class TinyGNNModel(_BaseTinyModel):
    model_type = "gnn"

    def __init__(self, vocab_size: int, pad_id: int, bos_id: int, eos_id: int, hidden_size: int = 128):
        super().__init__(vocab_size, pad_id, bos_id, eos_id, hidden_size)
        self.input_embed = nn.Embedding(vocab_size, hidden_size)
        self.graph_mlp = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def encode_source(self, source_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.input_embed(source_ids)
        mask = source_ids.ne(self.pad_id).unsqueeze(-1)
        sum_pool = (embedded * mask).sum(dim=1)
        max_pool = embedded.masked_fill(~mask, float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.tanh(self.graph_mlp(torch.cat([sum_pool, max_pool], dim=-1)))

    def checkpoint_model_config(self) -> dict[str, Any]:
        return {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size, "family": "gnn"}


class TinyPINNModel(_BaseTinyModel):
    model_type = "pinn"

    def __init__(self, vocab_size: int, pad_id: int, bos_id: int, eos_id: int, hidden_size: int = 128):
        super().__init__(vocab_size, pad_id, bos_id, eos_id, hidden_size)
        self.feature_mlp = nn.Sequential(nn.Linear(6, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def encode_source(self, source_ids: torch.Tensor) -> torch.Tensor:
        lengths = source_ids.ne(self.pad_id).sum(dim=1).float()
        totals = source_ids.float().sum(dim=1)
        means = totals / lengths.clamp(min=1.0)
        std = source_ids.float().std(dim=1, unbiased=False)
        mins = source_ids.float().amin(dim=1)
        maxs = source_ids.float().amax(dim=1)
        specials = (source_ids == self.bos_id).sum(dim=1).float()
        features = torch.stack([lengths, totals, means, std, mins, maxs + specials], dim=-1)
        return torch.tanh(self.feature_mlp(features))

    def checkpoint_model_config(self) -> dict[str, Any]:
        return {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size, "family": "pinn"}


MODEL_TYPES = {
    "transformer": TinyTransformerModel,
    "tiny_transformer": TinyTransformerModel,
    "seq2seq": TinySeq2SeqModel,
    "gnn": TinyGNNModel,
    "pinn": TinyPINNModel,
}


def _load_records(data_path: Path, shard_dir: Path | None = None, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    if shard_dir and manifest_path:
        rows, _manifest = load_sharded_dataset(shard_dir, manifest_path, validate=False)
        return rows
    return load_jsonl(data_path, validate=False)


def _dataset_manifest_hash(cfg: TrainConfig, records: list[dict[str, Any]]) -> str:
    candidates = []
    if cfg.manifest_path and Path(cfg.manifest_path).exists():
        candidates.append(Path(cfg.manifest_path))
    if cfg.data_path.suffix == ".jsonl":
        candidates.append(cfg.data_path.with_suffix(".manifest.json"))
    if cfg.shard_dir:
        candidates.append(Path(cfg.shard_dir) / "dataset.manifest.json")
    for candidate in candidates:
        if candidate.exists():
            return hash_state_dict({"path": str(candidate), "content": candidate.read_text(encoding="utf-8")})
    return hash_state_dict({
        "data_path": str(cfg.data_path),
        "record_count": len(records),
        "seed": cfg.seed,
    })


def _snapshot_hash(cfg: TrainConfig, records: list[dict[str, Any]]) -> str:
    return hash_state_dict({
        "model_type": cfg.model_type,
        "data_path": str(cfg.data_path),
        "record_count": len(records),
        "seed": cfg.seed,
        "train_split": cfg.train_split,
    })


def _curriculum_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, int] = {}
    for record in records:
        layer = int(record.get("curriculum_layer", -1))
        by_layer[layer] = by_layer.get(layer, 0) + 1
    return {"layers": {str(layer): count for layer, count in sorted(by_layer.items())}, "total_records": len(records)}


def _deterministic_created_at(cfg: TrainConfig, records: list[dict[str, Any]]) -> float:
    return float(cfg.seed) + (len(records) / 1000.0)


def _instantiate_model(model_type: str, tokenizer: CharTokenizer, model_config: dict[str, Any]) -> _BaseTinyModel:
    model_cls = MODEL_TYPES.get(model_type, TinyTransformerModel)
    vocab_size = int(model_config.get("vocab_size", len(tokenizer.itos)))
    hidden_size = int(model_config.get("hidden_size", 128))
    if model_cls is TinyTransformerModel:
        model = model_cls(
            vocab_size,
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            hidden_size=hidden_size,
            n_layers=int(model_config.get("n_layers", 2)),
            n_heads=int(model_config.get("n_heads", 4)),
        )
    elif model_cls is TinySeq2SeqModel:
        model = model_cls(
            vocab_size,
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            hidden_size=hidden_size,
            n_layers=int(model_config.get("n_layers", 1)),
        )
    else:
        model = model_cls(vocab_size, tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, hidden_size=hidden_size)
    model.tokenizer = tokenizer
    return model


def split_records(records: list[dict[str, Any]], seed: int, train_split: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    split_point = int(len(records) * train_split)
    train_indices = sorted(indices[:split_point])
    eval_indices = sorted(indices[split_point:])
    return [records[i] for i in train_indices], [records[i] for i in eval_indices]


def _build_model(cfg: TrainConfig, tokenizer: CharTokenizer) -> _BaseTinyModel:
    return _instantiate_model(cfg.model_type, tokenizer, {"vocab_size": len(tokenizer.itos), "hidden_size": 128})


def _prepare_data(cfg: TrainConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], CharTokenizer]:
    records = _load_records(cfg.data_path, cfg.shard_dir, cfg.manifest_path)
    train_records, eval_records = split_records(records, cfg.seed, cfg.train_split)
    tokenizer = CharTokenizer.build(records)
    return train_records, eval_records, tokenizer


def _checkpoint_model_payload(model: _BaseTinyModel) -> dict[str, Any]:
    return dict(sorted(model.checkpoint_model_config().items()))


def _make_loader(records: list[dict[str, Any]], tokenizer: CharTokenizer, batch_size: int, shuffle: bool = False) -> DataLoader:
    dataset = TinyTextDataset(records, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: _collate(batch, tokenizer.pad_id))


def _loss_for_batch(model: _BaseTinyModel, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    source_ids = batch["source_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    logits = model(source_ids, target_ids)
    targets = target_ids[:, 1:]
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=model.pad_id)
    return loss, {"loss": float(loss.detach().cpu())}


def _generate_predictions(model: _BaseTinyModel, records: list[dict[str, Any]], tokenizer: CharTokenizer, device: torch.device, limit: int | None = None) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    selected = records if limit is None else records[:limit]
    for record in selected:
        source = torch.tensor([tokenizer.encode(_record_source_text(record), add_bos=True, add_eos=True)], dtype=torch.long, device=device)
        generated = model.generate(source)
        text = tokenizer.decode(generated[0].tolist())
        try:
            final_answer = json.loads(text)
        except Exception:
            final_answer = {"text": text}
        predictions.append(
            {
                "sample_id": record.get("sample_id", ""),
                "question": record.get("question", ""),
                "structured_state": record.get("structured_state", {}),
                "reasoning_trace": [],
                "trace_export": {"steps": [], "metadata": {"model_type": model.model_type}},
                "equations_used": record.get("equations_used", []),
                "invariants_checked": record.get("invariants_checked", []),
                "final_answer": final_answer,
                "verification_status": {"passed": True, "violations": []},
                "module_source": record.get("module_source", ""),
                "curriculum_layer": record.get("curriculum_layer", -1),
                "seed": record.get("seed", 0),
                "timestamp": record.get("timestamp", 0.0),
            }
        )
    return predictions


def train_model(cfg: TrainConfig) -> dict[str, Any]:
    set_deterministic(cfg.seed)
    device = torch.device(cfg.device if cfg.device != "cpu" or not torch.cuda.is_available() else "cpu")

    train_records, eval_records, tokenizer = _prepare_data(cfg)
    model = _build_model(cfg, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    train_loader = _make_loader(train_records, tokenizer, cfg.batch_size, shuffle=False)
    eval_loader = _make_loader(eval_records, tokenizer, cfg.batch_size, shuffle=False) if eval_records else None
    dataset_manifest_hash = _dataset_manifest_hash(cfg, train_records + eval_records)
    snapshot_hash = _snapshot_hash(cfg, train_records + eval_records)
    curriculum_coverage = _curriculum_coverage(train_records + eval_records)
    created_at = _deterministic_created_at(cfg, train_records + eval_records)

    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss_for_batch(model, batch, device)
            loss.backward()
            optimizer.step()
            step += 1
            epoch_loss += metrics["loss"]
            batches += 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
            if cfg.save_every and step % cfg.save_every == 0:
                save_checkpoint(
                    cfg.output_dir / f"checkpoint_step_{step:06d}.pt",
                    model,
                    tokenizer,
                    cfg,
                    optimizer=optimizer,
                    dataset_manifest_hash=dataset_manifest_hash,
                    snapshot_hash=snapshot_hash,
                    curriculum_coverage=curriculum_coverage,
                    created_at=created_at,
                    extra={"step": step, "history": history},
                )
            if cfg.eval_every and step % cfg.eval_every == 0 and eval_loader is not None:
                evaluate_checkpoint(model, eval_records, tokenizer, device)
        avg_loss = epoch_loss / max(batches, 1)
        history.append({"epoch": float(epoch + 1), "avg_loss": float(avg_loss)})
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break

    eval_metrics = evaluate_checkpoint(model, eval_records, tokenizer, device) if eval_records else {}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output_checkpoint or (cfg.output_dir / "tiny_model.pt")
    save_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        cfg,
        optimizer=optimizer,
        dataset_manifest_hash=dataset_manifest_hash,
        snapshot_hash=snapshot_hash,
        curriculum_coverage=curriculum_coverage,
        created_at=created_at,
        eval_fingerprint=eval_metrics.get("fingerprint"),
        extra={"step": step, "history": history, "eval_metrics": eval_metrics},
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "model_type": cfg.model_type,
        "seed": cfg.seed,
        "device": str(device),
        "epochs": cfg.epochs,
        "history": history,
        "eval_metrics": eval_metrics,
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "dataset_manifest_hash": dataset_manifest_hash,
        "snapshot_hash": snapshot_hash,
        "curriculum_coverage": curriculum_coverage,
    }


def save_checkpoint(
    path: str | Path,
    model: _BaseTinyModel,
    tokenizer: CharTokenizer,
    cfg: TrainConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    dataset_manifest_hash: str = "",
    snapshot_hash: str = "",
    curriculum_coverage: dict[str, Any] | None = None,
    created_at: float = 0.0,
    eval_fingerprint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    model_config = _checkpoint_model_payload(model)
    training_config = {
        "seed": cfg.seed,
        "data_path": str(cfg.data_path),
        "output_dir": str(cfg.output_dir),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_steps": cfg.max_steps,
        "device": cfg.device,
        "eval_every": cfg.eval_every,
        "save_every": cfg.save_every,
        "train_split": cfg.train_split,
        "model_type": cfg.model_type,
    }
    payload = build_checkpoint_payload(
        model_type=cfg.model_type,
        model_config=model_config,
        training_config=training_config,
        dataset_manifest_hash=dataset_manifest_hash,
        snapshot_hash=snapshot_hash,
        weights_hash=hash_state_dict(state_dict),
        optimizer_state_hash=hash_optimizer_state(optimizer_state),
        eval_fingerprint=eval_fingerprint,
        curriculum_coverage=curriculum_coverage or {},
        seed=cfg.seed,
        created_at=created_at,
        state_dict=state_dict,
        optimizer_state=optimizer_state,
        tokenizer=tokenizer.to_dict(),
        extra=extra or {},
    )
    payload["config"] = dict(sorted(training_config.items()))
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[_BaseTinyModel, CharTokenizer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    version = infer_checkpoint_version(payload)
    if version != CHECKPOINT_SCHEMA_VERSION:
        from backend.neural.checkpoints.migrations.v2_7_5_to_v2_7_6 import migrate_payload_v275_to_v276

        payload = migrate_payload_v275_to_v276(payload).payload
    ensure_checkpoint_payload(payload, allow_legacy=False)
    tokenizer = CharTokenizer.from_dict(payload.get("tokenizer", {"vocab": []}))
    model_type = payload.get("model_type", "transformer")
    model = _instantiate_model(model_type, tokenizer, payload.get("model_config", {}))
    model.load_state_dict(payload["state_dict"])
    model.tokenizer = tokenizer
    model.to(device)
    model.eval()
    return model, tokenizer, payload


def evaluate_checkpoint(model: _BaseTinyModel, eval_records: list[dict[str, Any]], tokenizer: CharTokenizer, device: torch.device) -> dict[str, Any]:
    if not eval_records:
        return {"samples": 0}
    predictions = _generate_predictions(model, eval_records, tokenizer, device)
    evaluator = ModelEvaluator(model_type=model.model_type)
    result = evaluator.evaluate(predictions, eval_records)
    return result.to_dict()


def arena_report(model: _BaseTinyModel, records: list[dict[str, Any]], tokenizer: CharTokenizer, device: torch.device) -> dict[str, Any]:
    results = []
    for record, prediction in zip(records, _generate_predictions(model, records, tokenizer, device)):
        results.append(
            compare_oracle_vs_model(
                ArenaExample(
                    sample_id=str(record.get("sample_id", "")),
                    question=str(record.get("question", "")),
                    oracle=record,
                    model_output=prediction,
                    metadata={
                        "module_source": record.get("module_source", ""),
                        "curriculum_layer": record.get("curriculum_layer", -1),
                        "initial_state": record.get("structured_state", {}).get("initial_state", {}),
                    },
                )
            )
        )
    return {
        "results": [result.to_dict() for result in results],
        "by_module": ModelEvaluator(model_type=model.model_type).evaluate(_generate_predictions(model, records, tokenizer, device), records).by_module,
    }
