"""Full Wiki-727K training with Qwen sentence vectors as DistilBERT inputs.

This file is intentionally self-contained so mighty-colab can transmit and run
it without cloning the private repository or forwarding credentials.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass(frozen=True)
class Config:
    output_dir: str = "/content/embeddibert_continuous"
    initial_husk: str = "/content/distilbert_interpretation_20k_husk.safetensors"
    dataset_repo: str = "TankNee/wiki-727k"
    dataset_revision: str = "deea53e4b00c63dc158dca4071a4e4a5a38e2934"
    base_model: str = "distilbert/distilbert-base-uncased"
    base_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    qwen_model: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    instruction: str = (
        "Instruct: Represent this Wikipedia sentence for topic segmentation, "
        "emphasizing features that indicate whether the following sentence begins "
        "a new topical section.\nQuery: "
    )
    qwen_output_dimension: int = 768
    qwen_max_length: int = 32768
    qwen_max_batch: int = 2048
    qwen_token_budget: int = 131072
    parquet_document_batch: int = 128
    epochs: int = 2
    train_batch_size: int = 4096
    evaluation_batch_size: int = 8192
    sentence_block_size: int = 65536
    encoder_learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    checkpoint_every_steps: int = 1000
    seed: int = 71


CONFIG = Config()
ROOT = Path(CONFIG.output_dir)
CACHE = ROOT / "sentence_cache"
CHECKPOINTS = ROOT / "checkpoints"
LOG_PATH = ROOT / "events.jsonl"


def emit(event: str, **values) -> None:
    record = {"event": event, "time": time.time(), **values}
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_exact(text: str, expected: int) -> list[str]:
    sentences = text.split("\n")
    if sentences and sentences[-1] == "":
        sentences.pop()
    if len(sentences) != expected:
        raise ValueError(f"Sentence/label mismatch: {len(sentences)} != {expected}")
    return sentences


def scan_parquet(path: Path) -> dict[str, int]:
    documents = sentences = positives = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["label"], batch_size=8192):
        labels = batch.column(0)
        offsets = labels.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        lengths = np.diff(offsets)
        values = labels.values.to_numpy(zero_copy_only=False).astype(np.int64)
        documents += len(labels)
        sentences += int(lengths.sum())
        positives += int(values.sum())
        final_indices = offsets[1:] - 1
        nonempty = lengths > 0
        if nonempty.any():
            positives -= int(values[final_indices[nonempty]].sum())
    return {
        "documents": documents,
        "sentences": sentences,
        "pairs": sentences - documents,
        "positives": positives,
        "negatives": sentences - documents - positives,
    }


@torch.inference_mode()
def encode_sentences(
    sentences: list[str],
    *,
    tokenizer,
    model,
    config: Config,
) -> tuple[np.ndarray, int]:
    prompts = [config.instruction + sentence for sentence in sentences]
    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=config.qwen_max_length,
        add_special_tokens=True,
    )
    lengths = np.fromiter((len(row) for row in encoded["input_ids"]), np.int32)
    truncated = int((lengths >= config.qwen_max_length).sum())
    order = np.argsort(lengths, kind="stable")
    output = np.empty((len(sentences), config.qwen_output_dimension), dtype=np.float16)
    cursor = 0
    while cursor < len(order):
        max_length = int(lengths[order[cursor]])
        maximum = min(
            config.qwen_max_batch,
            max(1, config.qwen_token_budget // max(1, max_length)),
        )
        end = min(len(order), cursor + maximum)
        # Because rows are length-sorted, shrink if the last row would exceed
        # the padded-token budget.
        while end > cursor + 1:
            padded_length = int(lengths[order[end - 1]])
            if (end - cursor) * padded_length <= config.qwen_token_budget:
                break
            end -= 1
        indices = order[cursor:end]
        features = [
            {key: encoded[key][int(index)] for key in encoded}
            for index in indices
        ]
        batch = tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**batch).last_hidden_state
            pooled = hidden[:, -1, : config.qwen_output_dimension]
        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        output[indices] = pooled.cpu().numpy().astype(np.float16)
        cursor = end
    return output, truncated


def open_cache(split: str, metadata: dict[str, int], mode: str):
    split_dir = CACHE / split
    return (
        np.memmap(
            split_dir / "embeddings.f16",
            dtype=np.float16,
            mode=mode,
            shape=(metadata["sentences"], CONFIG.qwen_output_dimension),
        ),
        np.memmap(
            split_dir / "labels.u8",
            dtype=np.uint8,
            mode=mode,
            shape=(metadata["sentences"],),
        ),
        np.memmap(
            split_dir / "valid.u8",
            dtype=np.uint8,
            mode=mode,
            shape=(metadata["sentences"],),
        ),
    )


def build_split_cache(split: str, parquet_path: Path, tokenizer, qwen) -> dict:
    split_dir = CACHE / split
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.json"
    progress_path = split_dir / "progress.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = scan_parquet(parquet_path)
        metadata.update(
            {
                "split": split,
                "dataset_revision": CONFIG.dataset_revision,
                "instruction": CONFIG.instruction,
                "dimensions": CONFIG.qwen_output_dimension,
                "qwen_revision": CONFIG.qwen_revision,
                "status": "allocated",
                "truncated_sentences": 0,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metadata.get("status") == "complete":
        emit("cache_reused", split=split, **{key: metadata[key] for key in ("sentences", "pairs")})
        return metadata

    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"completed_batches": 0, "sentence_cursor": 0, "truncated_sentences": 0}
    )
    mode = "r+" if progress["completed_batches"] else "w+"
    embeddings, labels_store, valid_store = open_cache(split, metadata, mode)
    parquet = pq.ParquetFile(parquet_path)
    started = time.time()
    for batch_index, batch in enumerate(
        parquet.iter_batches(
            columns=["text", "label"], batch_size=CONFIG.parquet_document_batch
        )
    ):
        if batch_index < progress["completed_batches"]:
            continue
        sentence_rows: list[str] = []
        label_rows: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        for text, labels in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            sentences = split_exact(text, len(labels))
            sentence_rows.extend(sentences)
            label_array = np.asarray(labels, dtype=np.uint8)
            valid_array = np.ones(len(labels), dtype=np.uint8)
            if len(labels):
                valid_array[-1] = 0
            label_rows.append(label_array)
            valid_rows.append(valid_array)
        vectors, truncated = encode_sentences(
            sentence_rows, tokenizer=tokenizer, model=qwen, config=CONFIG
        )
        start = progress["sentence_cursor"]
        end = start + len(sentence_rows)
        embeddings[start:end] = vectors
        labels_store[start:end] = np.concatenate(label_rows)
        valid_store[start:end] = np.concatenate(valid_rows)
        embeddings.flush()
        labels_store.flush()
        valid_store.flush()
        progress = {
            "completed_batches": batch_index + 1,
            "sentence_cursor": end,
            "truncated_sentences": progress["truncated_sentences"] + truncated,
        }
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            emit(
                "cache_progress",
                split=split,
                documents=min(
                    metadata["documents"],
                    (batch_index + 1) * CONFIG.parquet_document_batch,
                ),
                sentences=end,
                sentences_per_second=end / max(time.time() - started, 1e-6),
                truncated=progress["truncated_sentences"],
            )
    if progress["sentence_cursor"] != metadata["sentences"]:
        raise RuntimeError(
            f"Incomplete {split} cache: {progress['sentence_cursor']} != {metadata['sentences']}"
        )
    if int(valid_store.sum()) != metadata["pairs"]:
        raise RuntimeError(f"Pair count mismatch in {split}")
    metadata["status"] = "complete"
    metadata["truncated_sentences"] = progress["truncated_sentences"]
    metadata["embedding_sha256"] = sha256(split_dir / "embeddings.f16")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    emit("cache_complete", split=split, **metadata)
    return metadata


def apply_husk(backbone, path: Path) -> None:
    state = load_file(str(path), device="cpu")
    for index, layer in enumerate(backbone.transformer.layer):
        prefix = f"layer.{index}."
        layer_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        layer.load_state_dict(layer_state, strict=True)


class ContinuousPairClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        dimension = backbone.config.dim
        self.pre_classifier = nn.Linear(4 * dimension, dimension)
        self.dropout = nn.Dropout(backbone.config.seq_classif_dropout)
        self.classifier = nn.Linear(dimension, 2)

    def forward(self, sentence_pairs: torch.Tensor) -> torch.Tensor:
        if sentence_pairs.ndim != 3 or sentence_pairs.shape[1:] != (2, 768):
            raise ValueError("sentence_pairs must have shape [batch, 2, 768]")
        mask = torch.ones(
            sentence_pairs.shape[:2], dtype=torch.long, device=sentence_pairs.device
        )
        hidden = self.backbone(
            inputs_embeds=sentence_pairs,
            attention_mask=mask,
            return_dict=True,
        ).last_hidden_state
        left, right = hidden[:, 0], hidden[:, 1]
        features = torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
        features = F.gelu(self.pre_classifier(features))
        return self.classifier(self.dropout(features))


def block_counts(valid: np.memmap, block_size: int) -> list[int]:
    return [
        int(valid[start : min(start + block_size, len(valid))].sum())
        for start in range(0, len(valid), block_size)
    ]


def pair_batches(split: str, metadata: dict, batch_size: int, epoch_seed: int, shuffle: bool):
    embeddings, labels, valid = open_cache(split, metadata, "r")
    starts = list(range(0, metadata["sentences"], CONFIG.sentence_block_size))
    rng = np.random.default_rng(epoch_seed)
    if shuffle:
        rng.shuffle(starts)
    for start in starts:
        end = min(start + CONFIG.sentence_block_size, metadata["sentences"])
        extra_end = min(end + 1, metadata["sentences"])
        block_embeddings = np.array(embeddings[start:extra_end], copy=True)
        local = np.flatnonzero(np.asarray(valid[start:end]))
        if shuffle:
            rng.shuffle(local)
        for cursor in range(0, len(local), batch_size):
            indices = local[cursor : cursor + batch_size]
            pairs = np.stack(
                (block_embeddings[indices], block_embeddings[indices + 1]), axis=1
            )
            targets = np.array(labels[start:end][indices], dtype=np.int64, copy=True)
            yield torch.from_numpy(pairs), torch.from_numpy(targets)


@torch.inference_mode()
def evaluate(model, split: str, metadata: dict, device: torch.device) -> dict[str, float]:
    model.eval()
    tp = fp = tn = fn = count = 0
    loss_sum = 0.0
    for pairs, targets in pair_batches(
        split,
        metadata,
        CONFIG.evaluation_batch_size,
        CONFIG.seed,
        shuffle=False,
    ):
        pairs = pairs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(pairs)
            loss = F.cross_entropy(logits.float(), targets)
        predictions = logits.argmax(dim=-1)
        tp += int(((predictions == 1) & (targets == 1)).sum())
        fp += int(((predictions == 1) & (targets == 0)).sum())
        tn += int(((predictions == 0) & (targets == 0)).sum())
        fn += int(((predictions == 0) & (targets == 1)).sum())
        count += targets.numel()
        loss_sum += float(loss) * targets.numel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "examples": count,
        "loss": loss_sum / count,
        "accuracy": (tp + tn) / count,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().to(torch.float16).contiguous()
        for key, value in model.state_dict().items()
    }


def save_training_checkpoint(model, optimizer, scheduler, epoch, global_step, block, result):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / "latest.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "next_block": block,
            "result": result,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        path,
    )
    emit("checkpoint", path=str(path), epoch=epoch, step=global_step, next_block=block)


def train(model, metadata: dict[str, dict], device: torch.device) -> dict:
    _, _, train_valid = open_cache("train", metadata["train"], "r")
    counts = block_counts(train_valid, CONFIG.sentence_block_size)
    steps_per_epoch = sum(math.ceil(count / CONFIG.train_batch_size) for count in counts)
    total_steps = steps_per_epoch * CONFIG.epochs
    head_parameters = list(model.pre_classifier.parameters()) + list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    encoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": CONFIG.encoder_learning_rate},
            {"params": head_parameters, "lr": CONFIG.head_learning_rate},
        ],
        weight_decay=CONFIG.weight_decay,
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * CONFIG.warmup_fraction),
        num_training_steps=total_steps,
    )
    class_weight = torch.tensor(
        [1.0, metadata["train"]["negatives"] / metadata["train"]["positives"]],
        device=device,
    )
    result = {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "class_weight": class_weight.cpu().tolist(),
        "epochs": [],
    }
    global_step = 0
    started = time.time()
    model.train()
    for epoch in range(CONFIG.epochs):
        running_loss = 0.0
        running_examples = 0
        for pairs, targets in pair_batches(
            "train",
            metadata["train"],
            CONFIG.train_batch_size,
            CONFIG.seed + epoch,
            shuffle=True,
        ):
            pairs = pairs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pairs)
                loss = F.cross_entropy(logits.float(), targets, weight=class_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += float(loss.detach()) * targets.numel()
            running_examples += targets.numel()
            if global_step == 1 or global_step % 100 == 0:
                emit(
                    "train",
                    epoch=epoch + 1,
                    step=global_step,
                    total_steps=total_steps,
                    loss=running_loss / running_examples,
                    examples=running_examples,
                    examples_per_second=running_examples / max(time.time() - started, 1e-6),
                    encoder_lr=scheduler.get_last_lr()[0],
                    head_lr=scheduler.get_last_lr()[1],
                )
            if global_step % CONFIG.checkpoint_every_steps == 0:
                save_training_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, 0, result
                )
        development = evaluate(model, "dev", metadata["dev"], device)
        epoch_result = {
            "epoch": epoch + 1,
            "training_loss": running_loss / running_examples,
            "training_examples": running_examples,
            "development": development,
        }
        result["epochs"].append(epoch_result)
        emit("epoch_complete", **epoch_result)
        save_training_checkpoint(
            model, optimizer, scheduler, epoch + 1, global_step, 0, result
        )
        model.train()
    result["test"] = evaluate(model, "test", metadata["test"], device)
    emit("test_complete", **result["test"])
    return result


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.cuda.manual_seed_all(CONFIG.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected A100, found {torch.cuda.get_device_name(0)}")
    free_gib = shutil.disk_usage("/content").free / 2**30
    if free_gib < 80:
        raise RuntimeError(f"At least 80 GiB free disk required, found {free_gib:.2f}")
    initial_husk = Path(CONFIG.initial_husk)
    if not initial_husk.is_file():
        raise FileNotFoundError(initial_husk)
    emit(
        "start",
        config=asdict(CONFIG),
        device=torch.cuda.get_device_name(0),
        device_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        free_disk_gib=free_gib,
        initial_husk_sha256=sha256(initial_husk),
    )

    parquet_paths = {
        split: Path(
            hf_hub_download(
                repo_id=CONFIG.dataset_repo,
                repo_type="dataset",
                revision=CONFIG.dataset_revision,
                filename=f"data/{split}.parquet",
            )
        )
        for split in ("train", "dev", "test")
    }
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.qwen_model,
        revision=CONFIG.qwen_revision,
        padding_side="left",
    )
    qwen = AutoModel.from_pretrained(
        CONFIG.qwen_model,
        revision=CONFIG.qwen_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    metadata = {
        split: build_split_cache(split, path, tokenizer, qwen)
        for split, path in parquet_paths.items()
    }
    del qwen, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    emit("qwen_released", free_device_gib=torch.cuda.mem_get_info()[0] / 2**30)

    backbone = AutoModel.from_pretrained(
        CONFIG.base_model,
        revision=CONFIG.base_revision,
        attn_implementation="sdpa",
    )
    apply_husk(backbone, initial_husk)
    del backbone.embeddings.word_embeddings
    model = ContinuousPairClassifier(backbone).to(device)
    if any("word_embeddings" in name for name, _ in model.named_parameters()):
        raise RuntimeError("DistilBERT word embeddings were not removed")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    emit(
        "model_ready",
        trainable_parameters=total_parameters,
        word_embedding_parameters=0,
        input_contract="two Qwen sentence embeddings via inputs_embeds",
    )
    training = train(model, metadata, device)
    final_path = ROOT / "continuous_pair_distilbert.safetensors"
    save_file(
        model_state(model),
        final_path,
        metadata={
            "input_contract": "[batch,2,768] Qwen sentence embeddings",
            "qwen_revision": CONFIG.qwen_revision,
            "dataset_revision": CONFIG.dataset_revision,
        },
    )
    result = {
        "status": "complete",
        "config": asdict(CONFIG),
        "metadata": metadata,
        "training": training,
        "model": {
            "path": str(final_path),
            "sha256": sha256(final_path),
            "bytes": final_path.stat().st_size,
            "parameters": total_parameters,
            "contains_word_embedding_table": False,
        },
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }
    (ROOT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    emit("complete", result_path=str(ROOT / "result.json"), model=result["model"])


if __name__ == "__main__":
    main()
