from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer, BertModel

from .alignment import evaluate, make_first_layer_modules, train_stages
from .config import ExperimentConfig
from .embedding_table import build_qwen_embedding_table, save_embedding_table, table_statistics


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _module_state(prefix: str, module, output: dict[str, torch.Tensor]) -> None:
    for key, value in module.state_dict().items():
        output[f"{prefix}.{key}"] = value.detach().cpu().contiguous()


def run_experiment(config: ExperimentConfig) -> dict:
    started = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, config.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "config": config.to_dict(),
    }
    _json_dump(output_dir / "environment.json", environment)
    _json_dump(output_dir / "config.json", config.to_dict())

    bert_tokenizer = AutoTokenizer.from_pretrained(
        config.bert_model, revision=config.bert_revision, use_fast=True
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_model, revision=config.qwen_revision, padding_side="left"
    )
    qwen = AutoModel.from_pretrained(
        config.qwen_model,
        revision=config.qwen_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    table_result = build_qwen_embedding_table(
        bert_tokenizer=bert_tokenizer,
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

    bert = BertModel.from_pretrained(
        config.bert_model,
        revision=config.bert_revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    original_table = bert.embeddings.word_embeddings.weight.detach().float().cpu()
    if table_result.table.shape[0] < original_table.shape[0]:
        # Smoke runs replace a prefix and retain remaining teacher rows so module
        # shapes stay real. The output manifest labels this explicitly.
        replacement = original_table.clone()
        replacement[: table_result.table.shape[0]] = table_result.table
    else:
        replacement = table_result.table
    modules = make_first_layer_modules(bert, replacement)
    # Dataclasses are not nn.Modules; move each contained module explicitly.
    for module in vars(modules).values():
        module.to(device)
    del bert

    eval_batches = max(1, config.eval_examples // config.eval_batch_size)
    before = evaluate(
        modules, bert_tokenizer, batches=eval_batches,
        batch_size=config.eval_batch_size, sequence_length=config.sequence_length,
        seed=config.seed + 10_000, device=device
    )
    metrics_path = output_dir / "training_metrics.jsonl"

    def log(record: dict) -> None:
        record = {**record, "elapsed_seconds": time.time() - started}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    train_stages(
        modules,
        bert_tokenizer,
        sequence_length=config.sequence_length,
        batch_size=config.train_batch_size,
        seed=config.seed,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        attention_steps=config.attention_steps,
        intermediate_steps=config.intermediate_steps,
        output_steps=config.output_steps,
        attention_profile_weight=config.attention_profile_weight,
        hidden_state_weight=config.hidden_state_weight,
        log_every=config.log_every,
        device=device,
        log=log,
    )
    after = evaluate(
        modules, bert_tokenizer, batches=eval_batches,
        batch_size=config.eval_batch_size, sequence_length=config.sequence_length,
        seed=config.seed + 10_000, device=device
    )

    state: dict[str, torch.Tensor] = {"word_embeddings": replacement.to(torch.float16)}
    _module_state("attention", modules.student_attention, state)
    _module_state("intermediate", modules.student_intermediate, state)
    _module_state("output", modules.student_output, state)
    save_file(state, output_dir / "first_layer_husk.safetensors")

    result = {
        "status": "complete",
        "scope": "first_layer",
        "partial_vocabulary_smoke": table_result.table.shape[0] < len(bert_tokenizer),
        "qwen_table": table_statistics(table_result.table),
        "original_bert_table": table_statistics(original_table),
        "before": before,
        "after": after,
        "improvement": {
            key: (before[key] - after[key]) / max(abs(before[key]), 1e-12)
            for key in before
        },
        "elapsed_seconds": time.time() - started,
    }
    _json_dump(output_dir / "result.json", result)
    return result
