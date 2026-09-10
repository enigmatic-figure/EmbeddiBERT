"""Evaluate instruction-conditioned Qwen vectors as a DistilBERT steering channel.

The script intentionally performs inference only. It builds a fixed document-level
Wiki-727K sample, caches one sentence-vector table per distinct instruction, and
then evaluates the frozen continuous-input DistilBERT checkpoint for every
condition in an instruction manifest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DISTILBERT_MODEL = "distilbert/distilbert-base-uncased"
DISTILBERT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
WIKI727_DATASET = "TankNee/wiki-727k"
WIKI727_REVISION = "deea53e4b00c63dc158dca4071a4e4a5a38e2934"
DIMENSION = 768
CONDITION_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
EXPECTED_DEV_PARQUET_SHA256 = (
    "eb6eb83e96ae6e520e6038257369a0cb4b9d5e189a3f3eadb55206ab1a57b396"
)
EXPECTED_STUDENT_SHA256 = (
    "9ca37201a77f29cc53aaa55817e90e81c10a18bd629d91ea3eb6654801d088b7"
)
EXPECTED_DOCUMENTS = 500
EXPECTED_SENTENCES = 21634
EXPECTED_PAIRS = 21134
EXPECTED_BOUNDARIES = 2231
EXPECTED_SAMPLE_DIGEST = (
    "266660340abd2a5ef21146d5d8df9beada349e58aabd85fd885dfe2ad2b0fba7"
)
ENCODER_CONTRACT_VERSION = (
    "instruction-steering-cache-v2:qwen-bf16:sdpa:no-kv-cache:left-padding:"
    "pad-multiple-8:last-token:first-768:l2-fp32:store-fp16"
)


@dataclass(frozen=True)
class Condition:
    id: str
    description: str
    left_instruction: str
    right_instruction: str


@dataclass(frozen=True)
class DocumentSample:
    sentences: list[str]
    labels: np.ndarray
    pair_document_ids: np.ndarray
    sentence_offsets: np.ndarray
    pair_offsets: np.ndarray
    document_indices: np.ndarray
    digest: str


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(parts: Iterable[str | bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def split_exact(text: str, expected: int) -> list[str]:
    sentences = text.split("\n")
    if sentences and sentences[-1] == "":
        sentences.pop()
    if len(sentences) != expected:
        raise ValueError(f"Sentence/label mismatch: {len(sentences)} != {expected}")
    return sentences


def load_conditions(path: Path) -> tuple[str, str, list[Condition], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    round_id = str(payload["round_id"])
    anchor_id = str(payload["anchor_id"])
    conditions: list[Condition] = []
    seen: set[str] = set()
    for row in payload["conditions"]:
        condition_id = str(row["id"])
        if not CONDITION_ID.fullmatch(condition_id):
            raise ValueError(f"Invalid condition id: {condition_id!r}")
        if condition_id in seen:
            raise ValueError(f"Duplicate condition id: {condition_id}")
        seen.add(condition_id)
        instruction = row.get("instruction")
        left = row.get("left_instruction", instruction)
        right = row.get("right_instruction", instruction)
        if left is None or right is None:
            raise ValueError(
                f"Condition {condition_id} needs instruction or both role instructions"
            )
        conditions.append(
            Condition(
                id=condition_id,
                description=str(row.get("description", "")),
                left_instruction=str(left),
                right_instruction=str(right),
            )
        )
    if anchor_id not in seen:
        raise ValueError(f"Anchor condition {anchor_id!r} is absent")
    if not conditions:
        raise ValueError("At least one condition is required")
    return round_id, anchor_id, conditions, payload


def load_document_sample(parquet_path: Path, max_documents: int) -> DocumentSample:
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    sentences: list[str] = []
    label_rows: list[np.ndarray] = []
    pair_document_rows: list[np.ndarray] = []
    sentence_offsets = [0]
    pair_offsets = [0]
    document_indices: list[int] = []
    source_index = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=["text", "label"], batch_size=128):
        texts = batch.column(0).to_pylist()
        labels = batch.column(1).to_pylist()
        for text, label_values in zip(texts, labels):
            if len(document_indices) >= max_documents:
                break
            document_sentences = split_exact(text, len(label_values))
            pair_labels = np.asarray(label_values[:-1], dtype=np.uint8)
            document_id = len(document_indices)
            sentences.extend(document_sentences)
            label_rows.append(pair_labels)
            pair_document_rows.append(
                np.full(len(pair_labels), document_id, dtype=np.int32)
            )
            sentence_offsets.append(len(sentences))
            pair_offsets.append(pair_offsets[-1] + len(pair_labels))
            document_indices.append(source_index)
            source_index += 1
        if len(document_indices) >= max_documents:
            break
    if len(document_indices) != max_documents:
        raise ValueError(
            f"Requested {max_documents} documents but found {len(document_indices)}"
        )
    labels = np.concatenate(label_rows) if label_rows else np.empty(0, dtype=np.uint8)
    pair_document_ids = (
        np.concatenate(pair_document_rows)
        if pair_document_rows
        else np.empty(0, dtype=np.int32)
    )
    digest_parts: list[str | bytes] = []
    for document_id, (start, end) in enumerate(
        zip(sentence_offsets[:-1], sentence_offsets[1:])
    ):
        digest_parts.append(str(document_indices[document_id]))
        digest_parts.extend(sentences[start:end])
        digest_parts.append(label_rows[document_id].tobytes())
    return DocumentSample(
        sentences=sentences,
        labels=labels,
        pair_document_ids=pair_document_ids,
        sentence_offsets=np.asarray(sentence_offsets, dtype=np.int64),
        pair_offsets=np.asarray(pair_offsets, dtype=np.int64),
        document_indices=np.asarray(document_indices, dtype=np.int64),
        digest=stable_digest(digest_parts),
    )


def validate_canonical_artifacts(
    *,
    max_documents: int,
    parquet_sha256: str,
    student_sha256: str,
    source_commit: str,
) -> None:
    if max_documents != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Canonical evaluation requires {EXPECTED_DOCUMENTS} documents"
        )
    if parquet_sha256 != EXPECTED_DEV_PARQUET_SHA256:
        raise ValueError(
            "Canonical evaluation requires the pinned Wiki-727K dev parquet: "
            f"{parquet_sha256} != {EXPECTED_DEV_PARQUET_SHA256}"
        )
    if student_sha256 != EXPECTED_STUDENT_SHA256:
        raise ValueError(
            "Canonical evaluation requires the final continuous checkpoint: "
            f"{student_sha256} != {EXPECTED_STUDENT_SHA256}"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Canonical evaluation requires a 40-character source commit")


def validate_canonical_sample(sample: DocumentSample) -> None:
    observed = (
        len(sample.document_indices),
        len(sample.sentences),
        len(sample.labels),
        int(sample.labels.sum()),
        sample.digest,
    )
    expected = (
        EXPECTED_DOCUMENTS,
        EXPECTED_SENTENCES,
        EXPECTED_PAIRS,
        EXPECTED_BOUNDARIES,
        EXPECTED_SAMPLE_DIGEST,
    )
    if observed != expected:
        raise ValueError(f"Canonical sample contract mismatch: {observed!r} != {expected!r}")


def cache_identity(
    sample_digest: str,
    instruction: str,
    max_length: int,
    generator_fingerprint: str = "test-generator",
) -> str:
    return stable_digest(
        (
            ENCODER_CONTRACT_VERSION,
            QWEN_MODEL,
            QWEN_REVISION,
            str(DIMENSION),
            str(max_length),
            sample_digest,
            instruction,
            generator_fingerprint,
        )
    )


def emit(events_path: Path, event: str, **values: Any) -> None:
    record = {"event": event, "time": time.time(), **values}
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def select_batch_end(
    lengths: np.ndarray,
    order: np.ndarray,
    cursor: int,
    max_batch_size: int,
    token_budget: int,
) -> int:
    end = min(len(order), cursor + max_batch_size)
    while end > cursor + 1:
        padded_length = int(lengths[order[end - 1]])
        if (end - cursor) * padded_length <= token_budget:
            break
        end -= 1
    return end


def encode_instruction(
    sentences: list[str],
    instruction: str,
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    max_length: int,
    max_batch_size: int,
    token_budget: int,
    output_path: Path,
    metadata_path: Path,
    sample_digest: str,
    generator: dict[str, Any],
    generator_fingerprint: str,
    events_path: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    identity = cache_identity(
        sample_digest, instruction, max_length, generator_fingerprint
    )
    if output_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_bytes = len(sentences) * DIMENSION * np.dtype(np.float16).itemsize
        if (
            metadata.get("status") == "complete"
            and metadata.get("identity") == identity
            and output_path.stat().st_size == expected_bytes
            and metadata.get("vector_sha256") == sha256_file(output_path)
        ):
            emit(events_path, "embedding_cache_reused", identity=identity)
            return metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    vectors = np.memmap(
        temporary,
        dtype=np.float16,
        mode="w+",
        shape=(len(sentences), DIMENSION),
    )
    prompts = [instruction + sentence for sentence in sentences]
    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    lengths = np.fromiter((len(row) for row in encoded["input_ids"]), dtype=np.int32)
    truncated = int((lengths >= max_length).sum())
    order = np.argsort(lengths, kind="stable")
    runtime_budget = token_budget
    cursor = 0
    started = time.time()
    emit(
        events_path,
        "embedding_cache_started",
        identity=identity,
        sentences=len(sentences),
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
    while cursor < len(order):
        end = select_batch_end(
            lengths, order, cursor, max_batch_size, runtime_budget
        )
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
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(**batch, use_cache=False).last_hidden_state
                pooled = hidden[:, -1, :DIMENSION]
                pooled = functional.normalize(pooled.float(), p=2, dim=-1)
        except torch.OutOfMemoryError:
            failed_tokens = (end - cursor) * int(batch["input_ids"].shape[1])
            del batch
            torch.cuda.empty_cache()
            if end - cursor <= 1:
                raise
            runtime_budget = max(
                int(lengths[order[cursor]]),
                min(runtime_budget - 1, failed_tokens // 2),
            )
            emit(
                events_path,
                "qwen_budget_reduced",
                failed_padded_tokens=failed_tokens,
                new_token_budget=runtime_budget,
            )
            continue
        vectors[indices] = pooled.cpu().numpy().astype(np.float16)
        cursor = end
        if cursor == len(order) or cursor % 2048 < len(indices):
            vectors.flush()
            emit(
                events_path,
                "embedding_progress",
                identity=identity,
                completed=cursor,
                sentences=len(sentences),
                sentences_per_second=cursor / max(time.time() - started, 1e-6),
            )
        del batch, hidden, pooled
    vectors.flush()
    del vectors, encoded, prompts
    temporary.replace(output_path)
    metadata = {
        "status": "complete",
        "identity": identity,
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "sample_digest": sample_digest,
        "sentences": len(sentences),
        "dimension": DIMENSION,
        "max_length": max_length,
        "truncated_sentences": truncated,
        "runtime_token_budget": runtime_budget,
        "elapsed_seconds": time.time() - started,
        "bytes": output_path.stat().st_size,
        "vector_sha256": sha256_file(output_path),
        "encoder_contract": ENCODER_CONTRACT_VERSION,
        "generator": generator,
        "generator_fingerprint": generator_fingerprint,
    }
    write_json(metadata_path, metadata)
    emit(events_path, "embedding_cache_complete", **metadata)
    return metadata


def open_vectors(path: Path, sentence_count: int) -> np.memmap:
    return np.memmap(
        path,
        dtype=np.float16,
        mode="r",
        shape=(sentence_count, DIMENSION),
    )


def iter_pair_batches(
    left_vectors: np.memmap,
    right_vectors: np.memmap,
    sample: DocumentSample,
    batch_size: int,
):
    for document_id in range(len(sample.document_indices)):
        sentence_start = int(sample.sentence_offsets[document_id])
        sentence_end = int(sample.sentence_offsets[document_id + 1])
        pair_start = int(sample.pair_offsets[document_id])
        pair_end = int(sample.pair_offsets[document_id + 1])
        pair_count = pair_end - pair_start
        for local_start in range(0, pair_count, batch_size):
            local_end = min(local_start + batch_size, pair_count)
            left = np.asarray(
                left_vectors[
                    sentence_start + local_start : sentence_start + local_end
                ],
                dtype=np.float32,
            )
            right = np.asarray(
                right_vectors[
                    sentence_start + local_start + 1 : sentence_start + local_end + 1
                ],
                dtype=np.float32,
            )
            yield pair_start + local_start, np.stack((left, right), axis=1)


def rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    cumulative_tp = np.cumsum(ranked_labels == 1)
    cumulative_fp = np.cumsum(ranked_labels == 0)
    # Thresholds cannot split a group of examples with equal scores.
    boundaries = np.flatnonzero(
        np.concatenate((ranked_scores[:-1] != ranked_scores[1:], [True]))
    )
    tp = cumulative_tp[boundaries]
    fp = cumulative_fp[boundaries]
    positives = int(cumulative_tp[-1])
    negatives = int(cumulative_fp[-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positives, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    tpr = np.concatenate(([0.0], tp / max(positives, 1), [1.0]))
    fpr = np.concatenate(([0.0], fp / max(negatives, 1), [1.0]))
    return {
        "roc_auc": float(np.trapezoid(tpr, fpr)),
        "average_precision": float(
            np.sum(np.diff(np.concatenate(([0.0], recall))) * precision)
        ),
        "best_slice_f1": float(f1[best]),
        "best_slice_threshold": float(ranked_scores[boundaries[best]]),
        "best_slice_precision": float(precision[best]),
        "best_slice_recall": float(recall[best]),
    }


def calibration_metrics(
    labels: np.ndarray, scores: np.ndarray, bins: int = 15
) -> dict[str, float]:
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1 - 1e-7)
    labels_float = labels.astype(np.float64)
    bin_ids = np.minimum((clipped * bins).astype(np.int64), bins - 1)
    calibration_error = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        if selected.any():
            calibration_error += float(selected.mean()) * abs(
                float(clipped[selected].mean()) - float(labels_float[selected].mean())
            )
    positives = clipped[labels == 1]
    negatives = clipped[labels == 0]
    return {
        "brier": float(np.mean((clipped - labels_float) ** 2)),
        "log_loss": float(
            -np.mean(
                labels_float * np.log(clipped)
                + (1 - labels_float) * np.log(1 - clipped)
            )
        ),
        "expected_calibration_error_15": calibration_error,
        "mean_score": float(clipped.mean()),
        "score_std": float(clipped.std()),
        "positive_mean_score": float(positives.mean()),
        "negative_mean_score": float(negatives.mean()),
        "class_mean_separation": float(positives.mean() - negatives.mean()),
    }


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predictions = scores >= 0.5
    positive = labels == 1
    tp = int((predictions & positive).sum())
    fp = int((predictions & ~positive).sum())
    tn = int((~predictions & ~positive).sum())
    fn = int((~predictions & positive).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    result: dict[str, Any] = {
        "examples": len(labels),
        "positive_rate": float(positive.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    result.update(rank_metrics(labels, scores))
    result.update(calibration_metrics(labels, scores))
    return result


def score_shift(anchor: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    delta = scores.astype(np.float64) - anchor.astype(np.float64)
    anchor_prediction = anchor >= 0.5
    prediction = scores >= 0.5
    return {
        "mean_delta": float(delta.mean()),
        "mean_absolute_delta": float(np.abs(delta).mean()),
        "delta_std": float(delta.std()),
        "delta_p05": float(np.quantile(delta, 0.05)),
        "delta_p50": float(np.quantile(delta, 0.50)),
        "delta_p95": float(np.quantile(delta, 0.95)),
        "pearson_correlation": float(np.corrcoef(anchor, scores)[0, 1]),
        "decision_flip_rate": float((anchor_prediction != prediction).mean()),
        "negative_to_positive": int((~anchor_prediction & prediction).sum()),
        "positive_to_negative": int((anchor_prediction & ~prediction).sum()),
    }


def cosine_summary(
    anchor: np.memmap, vectors: np.memmap, chunk_size: int = 4096
) -> dict[str, float]:
    rows: list[np.ndarray] = []
    for start in range(0, len(anchor), chunk_size):
        end = min(start + chunk_size, len(anchor))
        anchor_chunk = np.asarray(anchor[start:end], dtype=np.float32)
        vector_chunk = np.asarray(vectors[start:end], dtype=np.float32)
        dot = np.sum(anchor_chunk * vector_chunk, axis=1)
        denominator = np.linalg.norm(anchor_chunk, axis=1) * np.linalg.norm(
            vector_chunk, axis=1
        )
        rows.append(dot / np.maximum(denominator, 1e-12))
    cosine = np.concatenate(rows)
    return {
        "mean": float(cosine.mean()),
        "std": float(cosine.std()),
        "p05": float(np.quantile(cosine, 0.05)),
        "p50": float(np.quantile(cosine, 0.50)),
        "p95": float(np.quantile(cosine, 0.95)),
    }


def make_model(checkpoint: Path, device: Any, *, offline: bool):
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file
    from torch import nn
    from transformers import AutoModel

    class ContinuousPairClassifier(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            self.backbone = backbone
            dimension = backbone.config.dim
            self.pre_classifier = nn.Linear(4 * dimension, dimension)
            self.dropout = nn.Dropout(backbone.config.seq_classif_dropout)
            self.classifier = nn.Linear(dimension, 2)

        def forward(self, pairs):
            if pairs.ndim != 3 or pairs.shape[1:] != (2, DIMENSION):
                raise ValueError(f"pairs must have shape [batch, 2, {DIMENSION}]")
            mask = torch.ones(pairs.shape[:2], dtype=torch.long, device=pairs.device)
            hidden = self.backbone(
                inputs_embeds=pairs, attention_mask=mask, return_dict=True
            ).last_hidden_state
            left, right = hidden[:, 0], hidden[:, 1]
            features = torch.cat(
                (left, right, torch.abs(left - right), left * right), dim=-1
            )
            return self.classifier(
                self.dropout(functional.gelu(self.pre_classifier(features)))
            )

    backbone = AutoModel.from_pretrained(
        DISTILBERT_MODEL,
        revision=DISTILBERT_REVISION,
        attn_implementation="sdpa",
        local_files_only=offline,
    )
    del backbone.embeddings.word_embeddings
    model = ContinuousPairClassifier(backbone)
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def score_condition(
    model: Any,
    left_vectors: np.memmap,
    right_vectors: np.memmap,
    sample: DocumentSample,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    import torch

    scores = np.empty(len(sample.labels), dtype=np.float32)
    with torch.inference_mode():
        for offset, pairs in iter_pair_batches(
            left_vectors, right_vectors, sample, batch_size
        ):
            tensor = torch.from_numpy(pairs).to(
                device=device, dtype=torch.float16, non_blocking=True
            )
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(tensor)
            batch_scores = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            scores[offset : offset + len(batch_scores)] = batch_scores
    return scores


def environment_record(torch_module: Any) -> dict[str, Any]:
    import transformers

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_module.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch_module.cuda.get_device_name(0),
        "cuda_memory_bytes": torch_module.cuda.get_device_properties(0).total_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--qwen-batch-size", type=int, default=128)
    parser.add_argument("--qwen-token-budget", type=int, default=32768)
    parser.add_argument("--student-batch-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--allow-noncanonical-sample", action="store_true")
    args = parser.parse_args()

    round_id, anchor_id, conditions, manifest = load_conditions(args.conditions)
    conditions_sha256 = sha256_file(args.conditions)
    round_dir = args.output_dir / round_id
    cache_dir = args.output_dir / "embedding_cache"
    events_path = round_dir / "events.jsonl"
    parquet_sha256 = sha256_file(args.parquet)
    student_sha256 = sha256_file(args.student)
    if not args.allow_noncanonical_sample:
        validate_canonical_artifacts(
            max_documents=args.max_documents,
            parquet_sha256=parquet_sha256,
            student_sha256=student_sha256,
            source_commit=args.source_commit,
        )
    sample = load_document_sample(args.parquet, args.max_documents)
    if not args.allow_noncanonical_sample:
        validate_canonical_sample(sample)
    sample_record = {
        "dataset": WIKI727_DATASET,
        "dataset_revision": WIKI727_REVISION,
        "parquet": str(args.parquet),
        "parquet_sha256": parquet_sha256,
        "canonical": not args.allow_noncanonical_sample,
        "documents": len(sample.document_indices),
        "sentences": len(sample.sentences),
        "pairs": len(sample.labels),
        "boundaries": int(sample.labels.sum()),
        "positive_rate": float(sample.labels.mean()),
        "first_document_index": int(sample.document_indices[0]),
        "last_document_index": int(sample.document_indices[-1]),
        "sample_digest": sample.digest,
    }
    round_dir.mkdir(parents=True, exist_ok=True)
    write_json(round_dir / "sample.json", sample_record)
    write_json(round_dir / "conditions.resolved.json", manifest)
    np.save(round_dir / "labels.npy", sample.labels)
    np.save(round_dir / "pair_document_ids.npy", sample.pair_document_ids)
    np.save(round_dir / "sentence_offsets.npy", sample.sentence_offsets)
    np.save(round_dir / "pair_offsets.npy", sample.pair_offsets)
    if args.inspect_only:
        print(json.dumps(sample_record, indent=2))
        return

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.qwen_token_budget <= 0 or args.qwen_batch_size <= 0:
        raise ValueError("Qwen batch size and token budget must be positive")
    device = torch.device("cuda")
    generator = {
        "encoder_contract": ENCODER_CONTRACT_VERSION,
        "source_commit": args.source_commit,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }
    generator_fingerprint = stable_digest(
        (json.dumps(generator, sort_keys=True, separators=(",", ":")),)
    )
    emit(
        events_path,
        "round_started",
        round_id=round_id,
        sample=sample_record,
        conditions=len(conditions),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL,
        revision=QWEN_REVISION,
        padding_side="left",
        local_files_only=args.offline,
    )
    qwen = AutoModel.from_pretrained(
        QWEN_MODEL,
        revision=QWEN_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=args.offline,
    ).to(device).eval()
    qwen.config.use_cache = False

    cache_records: dict[str, dict[str, Any]] = {}
    instruction_paths: dict[str, Path] = {}
    unique_instructions = list(
        dict.fromkeys(
            instruction
            for condition in conditions
            for instruction in (
                condition.left_instruction,
                condition.right_instruction,
            )
        )
    )
    for instruction in unique_instructions:
        identity = cache_identity(
            sample.digest,
            instruction,
            args.max_length,
            generator_fingerprint,
        )
        vector_path = cache_dir / f"{identity}.f16"
        metadata_path = cache_dir / f"{identity}.json"
        cache_records[identity] = encode_instruction(
            sample.sentences,
            instruction,
            tokenizer=tokenizer,
            model=qwen,
            device=device,
            max_length=args.max_length,
            max_batch_size=args.qwen_batch_size,
            token_budget=args.qwen_token_budget,
            output_path=vector_path,
            metadata_path=metadata_path,
            sample_digest=sample.digest,
            generator=generator,
            generator_fingerprint=generator_fingerprint,
            events_path=events_path,
        )
        instruction_paths[instruction] = vector_path
    del qwen, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    emit(events_path, "qwen_released")

    model = make_model(args.student, device, offline=args.offline)
    all_scores: dict[str, np.ndarray] = {}
    condition_results: dict[str, dict[str, Any]] = {}
    anchor_condition = next(row for row in conditions if row.id == anchor_id)
    ordered_conditions = [anchor_condition] + [
        row for row in conditions if row.id != anchor_id
    ]
    for condition in ordered_conditions:
        left_vectors = open_vectors(
            instruction_paths[condition.left_instruction], len(sample.sentences)
        )
        right_vectors = open_vectors(
            instruction_paths[condition.right_instruction], len(sample.sentences)
        )
        started = time.time()
        scores = score_condition(
            model,
            left_vectors,
            right_vectors,
            sample,
            device,
            args.student_batch_size,
        )
        np.save(round_dir / f"scores.{condition.id}.npy", scores)
        all_scores[condition.id] = scores
        condition_results[condition.id] = {
            "condition": asdict(condition),
            "metrics": classification_metrics(sample.labels, scores),
            "elapsed_seconds": time.time() - started,
        }
        emit(
            events_path,
            "condition_scored",
            condition_id=condition.id,
            elapsed_seconds=condition_results[condition.id]["elapsed_seconds"],
            metrics=condition_results[condition.id]["metrics"],
        )

    anchor_scores = all_scores[anchor_id]
    anchor_left = open_vectors(
        instruction_paths[anchor_condition.left_instruction], len(sample.sentences)
    )
    anchor_right = open_vectors(
        instruction_paths[anchor_condition.right_instruction], len(sample.sentences)
    )
    for condition in ordered_conditions:
        left_vectors = open_vectors(
            instruction_paths[condition.left_instruction], len(sample.sentences)
        )
        right_vectors = open_vectors(
            instruction_paths[condition.right_instruction], len(sample.sentences)
        )
        condition_results[condition.id]["versus_anchor"] = {
            "score_shift": score_shift(anchor_scores, all_scores[condition.id]),
            "left_embedding_cosine": cosine_summary(anchor_left, left_vectors),
            "right_embedding_cosine": cosine_summary(anchor_right, right_vectors),
        }

    result = {
        "status": "complete",
        "round_id": round_id,
        "anchor_id": anchor_id,
        "source_commit": args.source_commit,
        "conditions_sha256": conditions_sha256,
        "sample": sample_record,
        "models": {
            "qwen": QWEN_MODEL,
            "qwen_revision": QWEN_REVISION,
            "distilbert": DISTILBERT_MODEL,
            "distilbert_revision": DISTILBERT_REVISION,
            "student_path": str(args.student),
            "student_sha256": student_sha256,
        },
        "parameters": {
            "max_documents": args.max_documents,
            "qwen_batch_size": args.qwen_batch_size,
            "qwen_token_budget": args.qwen_token_budget,
            "student_batch_size": args.student_batch_size,
            "max_length": args.max_length,
            "offline": args.offline,
            "allow_noncanonical_sample": args.allow_noncanonical_sample,
        },
        "generator": generator,
        "generator_fingerprint": generator_fingerprint,
        "environment": environment_record(torch),
        "embedding_caches": cache_records,
        "conditions": condition_results,
    }
    write_json(round_dir / "results.json", result)
    emit(events_path, "round_complete", result=str(round_dir / "results.json"))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
