from __future__ import annotations

import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .distil_alignment import _draw_batch, _tokenize_corpus
from .husk import apply_distilbert_husk
from .wiki727 import load_alignment_jsonl


@dataclass(frozen=True)
class MlmAlignmentConfig:
    base_model: str = "distilbert/distilbert-base-uncased"
    base_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    corpus_path: str = "data/wiki727_alignment_8192.jsonl"
    independent_husk_path: str = (
        "outputs/distil_independent/distilbert_independent_husk.safetensors"
    )
    output_dir: str = "outputs/distil_mlm_alignment"
    seed: int = 47
    sequence_length: int = 128
    train_batch_size: int = 8
    eval_batch_size: int = 12
    eval_batches: int = 16
    mask_probability: float = 0.15
    temperature: float = 2.0
    interpretation_max_steps: int = 2400
    whole_model_max_steps: int = 1200
    interpretation_learning_rate: float = 3e-4
    whole_model_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    eval_every: int = 100
    early_stopping_patience: int = 6
    minimum_improvement: float = 1e-4
    gradient_clip: float = 1.0
    use_amp: bool = True
    enforce_interpretation_plateau: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "MlmAlignmentConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        result = cls(**payload)
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
            "sequence_length",
            "train_batch_size",
            "eval_batch_size",
            "eval_batches",
            "interpretation_max_steps",
            "whole_model_max_steps",
            "eval_every",
            "early_stopping_patience",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.mask_probability < 1.0:
            raise ValueError("mask_probability must be between zero and one")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


INTERPRETATION_MARKERS = (
    "distilbert.embeddings.position_embeddings.",
    "distilbert.embeddings.LayerNorm.",
    ".sa_layer_norm.",
    ".output_layer_norm.",
    "vocab_transform.",
    "vocab_layer_norm.",
)


def configure_trainable_parameters(model, stage: str) -> dict[str, Any]:
    if stage not in {"interpretation", "whole_model"}:
        raise ValueError(f"Unknown training stage: {stage}")

    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if stage == "interpretation":
            enabled = name == "vocab_projector.bias" or any(
                marker in name for marker in INTERPRETATION_MARKERS
            )
        else:
            enabled = name != "distilbert.embeddings.word_embeddings.weight"
        parameter.requires_grad_(enabled)
        (trainable if enabled else frozen).append(name)

    # The decoder weight is tied to the Qwen-derived input table. It must remain
    # fixed in both stages or the model can undo the new embedding interface.
    model.distilbert.embeddings.word_embeddings.weight.requires_grad_(False)
    if model.vocab_projector.weight.requires_grad:
        raise RuntimeError("Tied vocabulary/Qwen table unexpectedly became trainable")
    return {
        "stage": stage,
        "trainable_names": sorted(trainable),
        "frozen_names": sorted(frozen),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def make_masked_batch(
    corpus: dict[str, torch.Tensor],
    *,
    batch_size: int,
    generator: torch.Generator,
    mask_token_id: int,
    special_token_ids: set[int],
    mask_probability: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = _draw_batch(
        corpus,
        batch_size=batch_size,
        generator=generator,
        device=torch.device("cpu"),
    )
    eligible = attention_mask.bool()
    for token_id in special_token_ids:
        eligible &= input_ids.ne(token_id)
    selected = torch.rand(input_ids.shape, generator=generator) < mask_probability
    selected &= eligible
    if not selected.any():
        positions = eligible.nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise ValueError("Batch contains no maskable tokens")
        selected[tuple(positions[0])] = True

    corrupted = input_ids.clone()
    draw = torch.rand(input_ids.shape, generator=generator)
    replace_with_mask = selected & (draw < 0.8)
    replace_with_random = selected & (draw >= 0.8) & (draw < 0.9)
    corrupted[replace_with_mask] = mask_token_id
    random_tokens = torch.randint(
        0,
        int(corpus["vocab_size"]),
        input_ids.shape,
        generator=generator,
    )
    corrupted[replace_with_random] = random_tokens[replace_with_random]
    return corrupted, attention_mask, selected


def prediction_logits(model, input_ids, attention_mask, selected):
    hidden = model.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state[selected]
    hidden = model.vocab_transform(hidden)
    hidden = model.activation(hidden)
    hidden = model.vocab_layer_norm(hidden)
    return model.vocab_projector(hidden)


def distillation_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    student = student_logits.float() / temperature
    teacher = teacher_logits.float() / temperature
    teacher_probabilities = F.softmax(teacher, dim=-1)
    loss = F.kl_div(
        F.log_softmax(student, dim=-1),
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature**2)
    agreement = (
        student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
    ).float().mean()
    entropy = -(teacher_probabilities * F.log_softmax(teacher, dim=-1)).sum(-1).mean()
    return loss, {
        "teacher_kl": float(loss.detach()),
        "top1_agreement": float(agreement.detach()),
        "teacher_entropy": float(entropy.detach()),
    }


def evaluate_teacher_response(
    teacher,
    student,
    corpus: dict[str, torch.Tensor],
    *,
    batches: int,
    batch_size: int,
    seed: int,
    mask_token_id: int,
    special_token_ids: set[int],
    mask_probability: float,
    temperature: float,
    teacher_device: torch.device,
    student_device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    teacher.eval()
    student.eval()
    generator = torch.Generator().manual_seed(seed)
    totals: dict[str, float] = {}
    total_masked = 0
    amp_enabled = use_amp and student_device.type == "cuda"
    with torch.no_grad():
        for _ in range(batches):
            ids, mask, selected = make_masked_batch(
                corpus,
                batch_size=batch_size,
                generator=generator,
                mask_token_id=mask_token_id,
                special_token_ids=special_token_ids,
                mask_probability=mask_probability,
            )
            teacher_ids = ids.to(teacher_device)
            teacher_mask = mask.to(teacher_device)
            teacher_selected = selected.to(teacher_device)
            student_ids = ids.to(student_device)
            student_mask = mask.to(student_device)
            student_selected = selected.to(student_device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                teacher_logits = prediction_logits(
                    teacher, teacher_ids, teacher_mask, teacher_selected
                )
                student_logits = prediction_logits(
                    student, student_ids, student_mask, student_selected
                )
            teacher_logits = teacher_logits.to(student_device)
            _, values = distillation_metrics(
                student_logits, teacher_logits, temperature
            )
            count = int(selected.sum())
            total_masked += count
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / total_masked for key, value in totals.items()} | {
        "masked_tokens": total_masked
    }


def _snapshot_trainable(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_snapshot(model, state: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            destination = named[name]
            destination.copy_(value.to(destination.device, destination.dtype))


def train_stage(
    teacher,
    student,
    corpus: dict[str, torch.Tensor],
    eval_corpus: dict[str, torch.Tensor],
    *,
    stage: str,
    max_steps: int,
    learning_rate: float,
    config: MlmAlignmentConfig,
    mask_token_id: int,
    special_token_ids: set[int],
    teacher_device: torch.device,
    student_device: torch.device,
    log: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    manifest = configure_trainable_parameters(student, stage)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = config.use_amp and student_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = torch.Generator().manual_seed(
        config.seed + (10_000 if stage == "interpretation" else 20_000)
    )
    best = evaluate_teacher_response(
        teacher,
        student,
        eval_corpus,
        batches=config.eval_batches,
        batch_size=config.eval_batch_size,
        seed=config.seed + 80_000,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
        mask_probability=config.mask_probability,
        temperature=config.temperature,
        teacher_device=teacher_device,
        student_device=student_device,
        use_amp=config.use_amp,
    )
    best_state = _snapshot_trainable(student)
    best_step = 0
    stale = 0
    evaluations = [{"step": 0, **best}]
    teacher.eval()
    student.train()

    for step in range(1, max_steps + 1):
        ids, mask, selected = make_masked_batch(
            corpus,
            batch_size=config.train_batch_size,
            generator=generator,
            mask_token_id=mask_token_id,
            special_token_ids=special_token_ids,
            mask_probability=config.mask_probability,
        )
        teacher_ids = ids.to(teacher_device)
        teacher_mask = mask.to(teacher_device)
        teacher_selected = selected.to(teacher_device)
        student_ids = ids.to(student_device)
        student_mask = mask.to(student_device)
        student_selected = selected.to(student_device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            teacher_logits = prediction_logits(
                teacher, teacher_ids, teacher_mask, teacher_selected
            )
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            student_logits = prediction_logits(
                student, student_ids, student_mask, student_selected
            )
            loss, values = distillation_metrics(
                student_logits, teacher_logits.to(student_device), config.temperature
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % max(1, config.eval_every // 5) == 0:
            log({"stage": stage, "step": step, "split": "train", **values})
        if step % config.eval_every != 0 and step != max_steps:
            continue

        current = evaluate_teacher_response(
            teacher,
            student,
            eval_corpus,
            batches=config.eval_batches,
            batch_size=config.eval_batch_size,
            seed=config.seed + 80_000,
            mask_token_id=mask_token_id,
            special_token_ids=special_token_ids,
            mask_probability=config.mask_probability,
            temperature=config.temperature,
            teacher_device=teacher_device,
            student_device=student_device,
            use_amp=config.use_amp,
        )
        evaluations.append({"step": step, **current})
        log({"stage": stage, "step": step, "split": "evaluation", **current})
        if current["teacher_kl"] < best["teacher_kl"] - config.minimum_improvement:
            best = current
            best_step = step
            best_state = _snapshot_trainable(student)
            stale = 0
        else:
            stale += 1
        student.train()
        if stale >= config.early_stopping_patience:
            break

    _restore_snapshot(student, best_state)
    student.eval()
    return {
        "manifest": manifest,
        "best_step": best_step,
        "best": best,
        "evaluations": evaluations,
        "stopped_early": evaluations[-1]["step"] < max_steps,
        "termination_reason": (
            "held_out_plateau"
            if evaluations[-1]["step"] < max_steps
            else "maximum_steps"
        ),
    }


def _save_stage(model, output_dir: Path, stage: str) -> dict[str, str]:
    stage_dir = output_dir / "layer_bank" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, torch.Tensor] = {
        "word_embeddings": model.distilbert.embeddings.word_embeddings.weight.detach()
        .cpu()
        .to(torch.float16)
        .contiguous()
    }
    for index, layer in enumerate(model.distilbert.transformer.layer):
        layer_state = {
            key: value.detach().cpu().to(torch.float16).contiguous()
            for key, value in layer.state_dict().items()
        }
        save_file(layer_state, stage_dir / f"layer_{index}.safetensors")
        state.update({f"layer.{index}.{key}": value for key, value in layer_state.items()})
    husk_path = output_dir / f"distilbert_{stage}_husk.safetensors"
    save_file(state, husk_path, metadata={"stage": stage})

    head_state = {
        name: parameter.detach().cpu().to(torch.float16).contiguous()
        for name, parameter in model.state_dict().items()
        if name.startswith(("vocab_transform.", "vocab_layer_norm."))
        or name == "vocab_projector.bias"
    }
    head_path = output_dir / f"distilbert_{stage}_mlm_head.safetensors"
    save_file(head_state, head_path, metadata={"stage": stage})
    return {"husk": str(husk_path), "mlm_head": str(head_path)}


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def run_mlm_alignment(config: MlmAlignmentConfig) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    student_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_device = torch.device(
        "cuda:1" if torch.cuda.device_count() > 1 else student_device
    )
    train_texts, eval_texts = load_alignment_jsonl(config.corpus_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.base_revision, use_fast=True
    )
    teacher = AutoModelForMaskedLM.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        attn_implementation="eager",
    ).to(teacher_device)
    student = AutoModelForMaskedLM.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        attn_implementation="eager",
    )
    apply_distilbert_husk(
        student.distilbert,
        config.independent_husk_path,
        load_word_embeddings=True,
    )
    student.tie_weights()
    student.to(student_device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    train_corpus = _tokenize_corpus(tokenizer, train_texts, config.sequence_length)
    eval_corpus = _tokenize_corpus(tokenizer, eval_texts, config.sequence_length)
    train_corpus["vocab_size"] = torch.tensor(len(tokenizer))
    eval_corpus["vocab_size"] = torch.tensor(len(tokenizer))
    special_token_ids = set(tokenizer.all_special_ids)
    metrics_path = output_dir / "training_metrics.jsonl"

    def log(record: dict[str, Any]) -> None:
        payload = {**record, "elapsed_seconds": time.time() - started}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)

    baseline = evaluate_teacher_response(
        teacher,
        student,
        eval_corpus,
        batches=config.eval_batches,
        batch_size=config.eval_batch_size,
        seed=config.seed + 80_000,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=special_token_ids,
        mask_probability=config.mask_probability,
        temperature=config.temperature,
        teacher_device=teacher_device,
        student_device=student_device,
        use_amp=config.use_amp,
    )
    interpretation = train_stage(
        teacher,
        student,
        train_corpus,
        eval_corpus,
        stage="interpretation",
        max_steps=config.interpretation_max_steps,
        learning_rate=config.interpretation_learning_rate,
        config=config,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=special_token_ids,
        teacher_device=teacher_device,
        student_device=student_device,
        log=log,
    )
    interpretation_artifacts = _save_stage(student, output_dir, "interpretation")
    _dump_json(
        output_dir / "partial_result.json",
        {"baseline": baseline, "interpretation": interpretation},
    )

    if config.enforce_interpretation_plateau and not interpretation["stopped_early"]:
        result = {
            "status": "interpretation_limit_reached",
            "objective": "masked_token_teacher_response_distillation",
            "baseline": baseline,
            "interpretation": interpretation,
            "whole_model": None,
            "artifacts": {"interpretation": interpretation_artifacts},
            "interface_anchor": (
                "Qwen-derived word/vocabulary table frozen; whole model not unlocked"
            ),
            "corpus": {
                "path": config.corpus_path,
                "train_spans": len(train_texts),
                "evaluation_spans": len(eval_texts),
            },
            "config": asdict(config),
            "elapsed_seconds": time.time() - started,
        }
        _dump_json(output_dir / "result.json", result)
        return result

    whole_model = train_stage(
        teacher,
        student,
        train_corpus,
        eval_corpus,
        stage="whole_model",
        max_steps=config.whole_model_max_steps,
        learning_rate=config.whole_model_learning_rate,
        config=config,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=special_token_ids,
        teacher_device=teacher_device,
        student_device=student_device,
        log=log,
    )
    whole_artifacts = _save_stage(student, output_dir, "whole_model")
    result = {
        "status": (
            "complete" if whole_model["stopped_early"] else "whole_model_limit_reached"
        ),
        "objective": "masked_token_teacher_response_distillation",
        "baseline": baseline,
        "interpretation": interpretation,
        "whole_model": whole_model,
        "artifacts": {
            "interpretation": interpretation_artifacts,
            "whole_model": whole_artifacts,
        },
        "interface_anchor": "Qwen-derived word/vocabulary table frozen in both stages",
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
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "student_device": str(student_device),
            "teacher_device": str(teacher_device),
        },
        "config": asdict(config),
        "elapsed_seconds": time.time() - started,
    }
    _dump_json(output_dir / "result.json", result)
    return result
