from __future__ import annotations

import copy
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

from .alignment import attention_kl, extended_attention_mask, masked_mse
from .embedding_table import (
    build_qwen_embedding_table,
    save_embedding_table,
    table_statistics,
)
from .wiki727 import load_alignment_jsonl


@dataclass(frozen=True)
class DistilAlignmentConfig:
    base_model: str = "distilbert/distilbert-base-uncased"
    base_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    qwen_model: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    corpus_path: str = "data/wiki727_alignment_8192.jsonl"
    output_dir: str = "outputs/distil_independent"
    seed: int = 29
    output_dimension: int = 768
    token_rendering: str = "surface"
    embedding_batch_size: int = 384
    max_vocab_tokens: int | None = None
    sequence_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 24
    eval_batches: int = 16
    attention_steps_per_layer: int = 400
    intermediate_steps_per_layer: int = 150
    output_steps_per_layer: int = 150
    attention_learning_rate: float = 2e-4
    projection_learning_rate: float = 2e-4
    weight_decay: float = 0.01
    attention_profile_weight: float = 1.0
    attention_hidden_weight: float = 1.0
    intermediate_preactivation_weight: float = 0.25
    log_every: int = 50
    qwen_dtype: str = "float16"

    @classmethod
    def from_json(cls, path: str | Path) -> "DistilAlignmentConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        result = cls(**payload)
        result.validate()
        return result

    def validate(self) -> None:
        if self.output_dimension != 768:
            raise ValueError("This DistilBERT experiment requires 768-dimensional inputs")
        if self.qwen_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported qwen_dtype")
        if self.token_rendering not in {"raw", "surface"}:
            raise ValueError("Unsupported token_rendering")
        for name in (
            "sequence_length",
            "train_batch_size",
            "eval_batch_size",
            "eval_batches",
            "embedding_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _tokenize_corpus(tokenizer, texts: list[str], sequence_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _draw_batch(
    corpus: dict[str, torch.Tensor],
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(
        0, corpus["input_ids"].shape[0], (batch_size,), generator=generator
    )
    return (
        corpus["input_ids"].index_select(0, indices).to(device),
        corpus["attention_mask"].index_select(0, indices).to(device),
    )


def _attention_state(layer, hidden: torch.Tensor, mask: torch.Tensor):
    raw, probabilities = layer.attention(
        hidden,
        attention_mask=extended_attention_mask(mask, hidden.dtype),
    )
    residual = layer.sa_layer_norm(raw + hidden)
    return raw, probabilities, residual


def _layer_metrics(teacher_layer, student_layer, teacher_hidden, student_hidden, mask):
    with torch.no_grad():
        _, teacher_probabilities, teacher_attention = _attention_state(
            teacher_layer, teacher_hidden, mask
        )
        _, student_probabilities, student_attention = _attention_state(
            student_layer, student_hidden, mask
        )
        teacher_pre = teacher_layer.ffn.lin1(teacher_attention)
        student_pre = student_layer.ffn.lin1(student_attention)
        teacher_up = teacher_layer.ffn.activation(teacher_pre)
        student_up = student_layer.ffn.activation(student_pre)
        teacher_down = teacher_layer.ffn.lin2(teacher_up)
        student_down = student_layer.ffn.lin2(student_up)
        teacher_block = teacher_layer.output_layer_norm(teacher_down + teacher_attention)
        student_block = student_layer.output_layer_norm(student_down + student_attention)
    return {
        "attention_kl": attention_kl(
            student_probabilities, teacher_probabilities, mask
        ).item(),
        "attention_hidden_mse": masked_mse(
            student_attention, teacher_attention, mask
        ).item(),
        "intermediate_preactivation_mse": masked_mse(student_pre, teacher_pre, mask).item(),
        "intermediate_activation_mse": masked_mse(student_up, teacher_up, mask).item(),
        "output_projection_mse": masked_mse(student_down, teacher_down, mask).item(),
        "block_output_mse": masked_mse(student_block, teacher_block, mask).item(),
    }


def evaluate_isolated_layer(
    teacher,
    student,
    layer_index: int,
    corpus: dict[str, torch.Tensor],
    *,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    totals: dict[str, float] = {}
    teacher_layer = teacher.transformer.layer[layer_index]
    student_layer = student.transformer.layer[layer_index]
    for _ in range(batches):
        input_ids, mask = _draw_batch(
            corpus, batch_size=batch_size, generator=generator, device=device
        )
        with torch.no_grad():
            teacher_hidden = teacher.embeddings(input_ids)
            student_hidden = student.embeddings(input_ids)
        metrics = _layer_metrics(
            teacher_layer, student_layer, teacher_hidden, student_hidden, mask
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / batches for key, value in totals.items()}


def evaluate_composition(
    teacher,
    student,
    corpus: dict[str, torch.Tensor],
    *,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    totals = {f"layer_{index}": 0.0 for index in range(len(teacher.transformer.layer))}
    with torch.no_grad():
        for _ in range(batches):
            input_ids, mask = _draw_batch(
                corpus, batch_size=batch_size, generator=generator, device=device
            )
            teacher_hidden = teacher.embeddings(input_ids)
            student_hidden = student.embeddings(input_ids)
            attention_mask = extended_attention_mask(mask, teacher_hidden.dtype)
            for index, (teacher_layer, student_layer) in enumerate(
                zip(teacher.transformer.layer, student.transformer.layer)
            ):
                teacher_hidden = teacher_layer(
                    teacher_hidden, attention_mask=attention_mask
                )
                student_hidden = student_layer(
                    student_hidden, attention_mask=attention_mask
                )
                totals[f"layer_{index}"] += masked_mse(
                    student_hidden, teacher_hidden, mask
                ).item()
    return {key: value / batches for key, value in totals.items()}


def _set_trainable(module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_independent_layer(
    teacher,
    student,
    layer_index: int,
    corpus: dict[str, torch.Tensor],
    *,
    config: DistilAlignmentConfig,
    device: torch.device,
    log: Callable[[dict[str, Any]], None],
) -> None:
    teacher_layer = teacher.transformer.layer[layer_index]
    student_layer = student.transformer.layer[layer_index]
    generator = torch.Generator().manual_seed(config.seed + 1000 * layer_index)

    stages = (
        (
            "attention",
            student_layer.attention,
            config.attention_steps_per_layer,
            config.attention_learning_rate,
        ),
        (
            "intermediate",
            student_layer.ffn.lin1,
            config.intermediate_steps_per_layer,
            config.projection_learning_rate,
        ),
        (
            "output",
            student_layer.ffn.lin2,
            config.output_steps_per_layer,
            config.projection_learning_rate,
        ),
    )
    for stage, target, steps, learning_rate in stages:
        _set_trainable(student_layer, False)
        _set_trainable(target, True)
        parameters = [parameter for parameter in target.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=config.weight_decay
        )
        for step in range(1, steps + 1):
            input_ids, mask = _draw_batch(
                corpus,
                batch_size=config.train_batch_size,
                generator=generator,
                device=device,
            )
            with torch.no_grad():
                teacher_hidden = teacher.embeddings(input_ids)
                student_hidden = student.embeddings(input_ids)
                _, teacher_probabilities, teacher_attention = _attention_state(
                    teacher_layer, teacher_hidden, mask
                )
            _, student_probabilities, student_attention = _attention_state(
                student_layer, student_hidden, mask
            )

            if stage == "attention":
                profile = attention_kl(
                    student_probabilities, teacher_probabilities, mask
                )
                hidden = masked_mse(student_attention, teacher_attention, mask)
                loss = (
                    config.attention_profile_weight * profile
                    + config.attention_hidden_weight * hidden
                )
                values = {"attention_kl": profile.item(), "hidden_mse": hidden.item()}
            elif stage == "intermediate":
                with torch.no_grad():
                    teacher_pre = teacher_layer.ffn.lin1(teacher_attention)
                    teacher_up = teacher_layer.ffn.activation(teacher_pre)
                student_pre = student_layer.ffn.lin1(student_attention.detach())
                student_up = student_layer.ffn.activation(student_pre)
                pre_loss = masked_mse(student_pre, teacher_pre, mask)
                activation_loss = masked_mse(student_up, teacher_up, mask)
                loss = (
                    config.intermediate_preactivation_weight * pre_loss
                    + activation_loss
                )
                values = {
                    "preactivation_mse": pre_loss.item(),
                    "activation_mse": activation_loss.item(),
                }
            else:
                with torch.no_grad():
                    teacher_pre = teacher_layer.ffn.lin1(teacher_attention)
                    teacher_up = teacher_layer.ffn.activation(teacher_pre)
                    teacher_down = teacher_layer.ffn.lin2(teacher_up)
                    student_pre = student_layer.ffn.lin1(student_attention)
                    student_up = student_layer.ffn.activation(student_pre)
                student_down = student_layer.ffn.lin2(student_up)
                loss = masked_mse(student_down, teacher_down, mask)
                values = {"projection_mse": loss.item()}

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {stage} loss at layer {layer_index}, step {step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == steps:
                log(
                    {
                        "layer": layer_index,
                        "stage": stage,
                        "step": step,
                        "loss": loss.item(),
                        **values,
                    }
                )
    _set_trainable(student_layer, False)


def _save_husk(student, output_dir: Path) -> None:
    state: dict[str, torch.Tensor] = {
        "word_embeddings": student.embeddings.word_embeddings.weight.detach()
        .cpu()
        .to(torch.float16)
        .contiguous()
    }
    bank_dir = output_dir / "layer_bank" / "independent"
    bank_dir.mkdir(parents=True, exist_ok=True)
    for index, layer in enumerate(student.transformer.layer):
        layer_state = {
            key: value.detach().cpu().to(torch.float16).contiguous()
            for key, value in layer.state_dict().items()
        }
        save_file(layer_state, bank_dir / f"layer_{index}.safetensors")
        state.update({f"layer.{index}.{key}": value for key, value in layer_state.items()})
    save_file(state, output_dir / "distilbert_independent_husk.safetensors")


def run_distil_alignment(config: DistilAlignmentConfig) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qwen_dtype = getattr(torch, config.qwen_dtype)
    if device.type == "cpu" and qwen_dtype == torch.float16:
        qwen_dtype = torch.float32

    train_texts, eval_texts = load_alignment_jsonl(config.corpus_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.base_revision, use_fast=True
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_model,
        revision=config.qwen_revision,
        padding_side="left",
    )
    qwen = AutoModel.from_pretrained(
        config.qwen_model,
        revision=config.qwen_revision,
        torch_dtype=qwen_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    table_result = build_qwen_embedding_table(
        bert_tokenizer=tokenizer,
        qwen_tokenizer=qwen_tokenizer,
        qwen_model=qwen,
        output_dimension=config.output_dimension,
        batch_size=config.embedding_batch_size,
        device=device,
        token_rendering=config.token_rendering,
        max_vocab_tokens=config.max_vocab_tokens,
    )
    save_embedding_table(table_result, output_dir)
    del qwen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = AutoModel.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    student = copy.deepcopy(teacher)
    if table_result.table.shape[0] < len(tokenizer):
        replacement = teacher.embeddings.word_embeddings.weight.detach().clone()
        replacement[: table_result.table.shape[0]] = table_result.table
    else:
        replacement = table_result.table
    with torch.no_grad():
        student.embeddings.word_embeddings.weight.copy_(replacement)
    teacher.config._attn_implementation = "eager"
    student.config._attn_implementation = "eager"
    teacher.eval().to(device)
    student.eval().to(device)
    _set_trainable(teacher, False)
    _set_trainable(student, False)

    train_corpus = _tokenize_corpus(tokenizer, train_texts, config.sequence_length)
    eval_corpus = _tokenize_corpus(tokenizer, eval_texts, config.sequence_length)
    metrics_path = output_dir / "training_metrics.jsonl"

    def log(record: dict[str, Any]) -> None:
        payload = {**record, "elapsed_seconds": time.time() - started}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)

    composition_before = evaluate_composition(
        teacher,
        student,
        eval_corpus,
        batches=config.eval_batches,
        batch_size=config.eval_batch_size,
        seed=config.seed + 90_000,
        device=device,
    )
    layers: dict[str, Any] = {}
    for index in range(len(teacher.transformer.layer)):
        before = evaluate_isolated_layer(
            teacher,
            student,
            index,
            eval_corpus,
            batches=config.eval_batches,
            batch_size=config.eval_batch_size,
            seed=config.seed + 100_000 + index,
            device=device,
        )
        train_independent_layer(
            teacher,
            student,
            index,
            train_corpus,
            config=config,
            device=device,
            log=log,
        )
        after = evaluate_isolated_layer(
            teacher,
            student,
            index,
            eval_corpus,
            batches=config.eval_batches,
            batch_size=config.eval_batch_size,
            seed=config.seed + 100_000 + index,
            device=device,
        )
        layers[str(index)] = {
            "before": before,
            "after": after,
            "relative_reduction": {
                key: (before[key] - after[key]) / max(abs(before[key]), 1e-12)
                for key in before
            },
        }
        _dump_json(output_dir / "partial_result.json", {"layers": layers})

    composition_after = evaluate_composition(
        teacher,
        student,
        eval_corpus,
        batches=config.eval_batches,
        batch_size=config.eval_batch_size,
        seed=config.seed + 90_000,
        device=device,
    )
    _save_husk(student, output_dir)
    result = {
        "status": "complete",
        "alignment_mode": "independent_causal_free",
        "layers": layers,
        "composition_before": composition_before,
        "composition_after": composition_after,
        "composition_relative_reduction": {
            key: (composition_before[key] - composition_after[key])
            / max(abs(composition_before[key]), 1e-12)
            for key in composition_before
        },
        "qwen_table": table_statistics(table_result.table),
        "partial_vocabulary_smoke": table_result.table.shape[0] < len(tokenizer),
        "corpus": {
            "path": config.corpus_path,
            "train_spans": len(train_texts),
            "evaluation_spans": len(eval_texts),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        },
        "config": asdict(config),
        "elapsed_seconds": time.time() - started,
    }
    _dump_json(output_dir / "result.json", result)
    return result
