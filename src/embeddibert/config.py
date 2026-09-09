from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    bert_model: str = "google-bert/bert-base-uncased"
    bert_revision: str = "86b5e0934494bd15c9632b12f734a8a67f723594"
    qwen_model: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    output_dir: str = "outputs/run"
    seed: int = 17
    output_dimension: int = 768
    embedding_batch_size: int = 384
    max_vocab_tokens: int | None = None
    token_rendering: str = "surface"
    sequence_length: int = 64
    train_batch_size: int = 16
    eval_batch_size: int = 32
    train_examples: int = 4096
    eval_examples: int = 512
    attention_steps: int = 1000
    intermediate_steps: int = 400
    output_steps: int = 400
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    attention_profile_weight: float = 1.0
    hidden_state_weight: float = 1.0
    log_every: int = 25
    save_full_model: bool = False
    dtype: str = "float16"

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.output_dimension <= 0:
            raise ValueError("output_dimension must be positive")
        if self.token_rendering not in {"raw", "surface"}:
            raise ValueError("token_rendering must be 'raw' or 'surface'")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        for name in (
            "embedding_batch_size",
            "sequence_length",
            "train_batch_size",
            "eval_batch_size",
            "train_examples",
            "eval_examples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

