from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .data import make_calibration_batch


@dataclass
class FirstLayerModules:
    teacher_embeddings: torch.nn.Module
    student_embeddings: torch.nn.Module
    teacher_attention: torch.nn.Module
    student_attention: torch.nn.Module
    teacher_intermediate: torch.nn.Module
    student_intermediate: torch.nn.Module
    teacher_output: torch.nn.Module
    student_output: torch.nn.Module


def make_first_layer_modules(bert_model, qwen_table: torch.Tensor) -> FirstLayerModules:
    layer = bert_model.encoder.layer[0]
    teacher_embeddings = copy.deepcopy(bert_model.embeddings)
    student_embeddings = copy.deepcopy(bert_model.embeddings)
    expected = student_embeddings.word_embeddings.weight.shape
    if qwen_table.shape != expected:
        raise ValueError(f"Replacement table shape {qwen_table.shape} != BERT shape {expected}")
    with torch.no_grad():
        target_dtype = student_embeddings.word_embeddings.weight.dtype
        student_embeddings.word_embeddings.weight.copy_(qwen_table.to(target_dtype))

    modules = FirstLayerModules(
        teacher_embeddings=teacher_embeddings,
        student_embeddings=student_embeddings,
        teacher_attention=copy.deepcopy(layer.attention),
        student_attention=copy.deepcopy(layer.attention),
        teacher_intermediate=copy.deepcopy(layer.intermediate),
        student_intermediate=copy.deepcopy(layer.intermediate),
        teacher_output=copy.deepcopy(layer.output),
        student_output=copy.deepcopy(layer.output),
    )
    # Recent Transformers releases default BERT to SDPA, which intentionally
    # returns no attention probabilities. Profile distillation requires the
    # explicit eager implementation.
    for attention in (modules.teacher_attention, modules.student_attention):
        if hasattr(attention.self, "config"):
            attention.self.config._attn_implementation = "eager"
    for name, module in vars(modules).items():
        module.eval()  # Disable dropout without disabling autograd.
        trainable = name.startswith("student_") and name != "student_embeddings"
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    # Keep LayerNorm fixed in this first attention experiment; train Q/K/V/O only.
    for parameter in modules.student_attention.output.LayerNorm.parameters():
        parameter.requires_grad_(False)
    return modules


def extended_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (1.0 - mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min


def masked_mse(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_token = (student.float() - teacher.float()).pow(2).mean(dim=-1)
    weights = mask.float()
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


def attention_kl(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    s = student.float().clamp_min(eps)
    t = teacher.float().clamp_min(eps)
    per_query = (t * (t.log() - s.log())).sum(dim=-1)
    query_mask = mask[:, None, :].float()
    return (per_query * query_mask).sum() / query_mask.sum().clamp_min(1.0) / student.shape[1]


def _attention_forward(module, hidden, mask):
    result = module(
        hidden_states=hidden,
        attention_mask=extended_attention_mask(mask, hidden.dtype),
        output_attentions=True,
    )
    return result[0], result[1]


def evaluate(
    modules: FirstLayerModules,
    tokenizer,
    *,
    batches: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    totals = {"attention_kl": 0.0, "attention_hidden_mse": 0.0,
              "intermediate_mse": 0.0, "output_dense_mse": 0.0}
    with torch.no_grad():
        for _ in range(batches):
            batch = make_calibration_batch(
                tokenizer, batch_size=batch_size, sequence_length=sequence_length,
                generator=rng, device=device
            )
            mask = batch["attention_mask"]
            embedding_args = {k: v for k, v in batch.items() if k in {"input_ids", "token_type_ids"}}
            teacher_embed = modules.teacher_embeddings(**embedding_args)
            student_embed = modules.student_embeddings(**embedding_args)
            teacher_attn, teacher_probs = _attention_forward(modules.teacher_attention, teacher_embed, mask)
            student_attn, student_probs = _attention_forward(modules.student_attention, student_embed, mask)
            teacher_up = modules.teacher_intermediate(teacher_attn)
            student_up = modules.student_intermediate(student_attn)
            # BertIntermediate includes GELU. BertOutput includes down projection,
            # dropout, residual, and LayerNorm; compare dense separately here.
            teacher_dense = modules.teacher_output.dense(teacher_up)
            student_dense = modules.student_output.dense(student_up)
            totals["attention_kl"] += attention_kl(student_probs, teacher_probs, mask).item()
            totals["attention_hidden_mse"] += masked_mse(student_attn, teacher_attn, mask).item()
            totals["intermediate_mse"] += masked_mse(student_up, teacher_up, mask).item()
            totals["output_dense_mse"] += masked_mse(student_dense, teacher_dense, mask).item()
    return {key: value / batches for key, value in totals.items()}


def train_stages(
    modules: FirstLayerModules,
    tokenizer,
    *,
    sequence_length: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    attention_steps: int,
    intermediate_steps: int,
    output_steps: int,
    attention_profile_weight: float,
    hidden_state_weight: float,
    log_every: int,
    device: torch.device,
    log: Callable[[dict], None],
) -> None:
    rng = random.Random(seed)

    def batch():
        return make_calibration_batch(
            tokenizer, batch_size=batch_size, sequence_length=sequence_length,
            generator=rng, device=device
        )

    stages = [
        ("attention", modules.student_attention, attention_steps),
        ("intermediate", modules.student_intermediate, intermediate_steps),
        ("output", modules.student_output.dense, output_steps),
    ]
    for stage, target, steps in stages:
        if steps <= 0:
            continue
        for module_name in ("student_attention", "student_intermediate", "student_output"):
            for parameter in getattr(modules, module_name).parameters():
                parameter.requires_grad_(False)
        for parameter in target.parameters():
            parameter.requires_grad_(True)
        if stage == "attention":
            for parameter in modules.student_attention.output.LayerNorm.parameters():
                parameter.requires_grad_(False)
        parameters = [p for p in target.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

        for step in range(1, steps + 1):
            item = batch()
            mask = item["attention_mask"]
            embedding_args = {k: v for k, v in item.items() if k in {"input_ids", "token_type_ids"}}
            with torch.no_grad():
                teacher_embed = modules.teacher_embeddings(**embedding_args)
                student_embed = modules.student_embeddings(**embedding_args)
                teacher_attn, teacher_probs = _attention_forward(
                    modules.teacher_attention, teacher_embed, mask
                )
            student_attn, student_probs = _attention_forward(
                modules.student_attention, student_embed, mask
            )

            if stage == "attention":
                profile = attention_kl(student_probs, teacher_probs, mask)
                hidden = masked_mse(student_attn, teacher_attn, mask)
                loss = attention_profile_weight * profile + hidden_state_weight * hidden
                metrics = {"attention_kl": profile.item(), "hidden_mse": hidden.item()}
            elif stage == "intermediate":
                with torch.no_grad():
                    teacher_up = modules.teacher_intermediate(teacher_attn)
                student_up = modules.student_intermediate(student_attn.detach())
                loss = masked_mse(student_up, teacher_up, mask)
                metrics = {"intermediate_mse": loss.item()}
            else:
                with torch.no_grad():
                    teacher_up = modules.teacher_intermediate(teacher_attn)
                    teacher_dense = modules.teacher_output.dense(teacher_up)
                    student_up = modules.student_intermediate(student_attn)
                student_dense = modules.student_output.dense(student_up)
                loss = masked_mse(student_dense, teacher_dense, mask)
                metrics = {"output_dense_mse": loss.item()}

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step == 1 or step % log_every == 0 or step == steps:
                log({"stage": stage, "step": step, "loss": loss.item(), **metrics})
