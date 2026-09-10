"""Evaluate the continuous Qwen-input model and the published baseline."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class ContinuousPairClassifier(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        dimension = backbone.config.dim
        self.pre_classifier = nn.Linear(4 * dimension, dimension)
        self.dropout = nn.Dropout(backbone.config.seq_classif_dropout)
        self.classifier = nn.Linear(dimension, 2)

    def forward(self, sentence_pairs: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(sentence_pairs.shape[:2], dtype=torch.long, device=sentence_pairs.device)
        hidden = self.backbone(inputs_embeds=sentence_pairs, attention_mask=mask, return_dict=True).last_hidden_state
        left, right = hidden[:, 0], hidden[:, 1]
        features = torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
        return self.classifier(F.gelu(self.pre_classifier(features)))


def metrics(logits: torch.Tensor, labels: torch.Tensor, loss_sum: float, count: int) -> dict:
    pred = logits.argmax(-1)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"examples": count, "loss": loss_sum / count, "accuracy": (tp + tn) / count,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def score_diagnostics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    tp = np.cumsum(ranked_labels == 1)
    fp = np.cumsum(ranked_labels == 0)
    positives = int(tp[-1])
    negatives = int(fp[-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positives, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    tpr = np.concatenate(([0.0], tp / max(positives, 1), [1.0]))
    fpr = np.concatenate(([0.0], fp / max(negatives, 1), [1.0]))
    return {
        "positive_rate": positives / len(labels),
        "roc_auc": float(np.trapezoid(tpr, fpr)),
        "average_precision": float(precision[ranked_labels == 1].mean()),
        "best_f1": float(f1[best]),
        "best_f1_threshold": float(scores[order[best]]),
        "best_f1_precision": float(precision[best]),
        "best_f1_recall": float(recall[best]),
    }


def load_rows(parquet_path: Path) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(parquet_path, columns=["text", "label"])
    texts = table.column("text").to_pylist()
    lengths = np.asarray([len(value) for value in table.column("label").to_pylist()], dtype=np.int64)
    return texts, lengths


@torch.inference_mode()
def eval_student(model, embeddings: np.memmap, labels: np.ndarray, valid: np.ndarray, lengths: np.ndarray, device: torch.device, batch_size: int) -> dict:
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    cursor = 0
    for length in lengths:
        if length <= 0:
            continue
        sentence_count = int(length)
        seq = np.asarray(embeddings[cursor:cursor + sentence_count], dtype=np.float32)
        cursor += sentence_count
        for start in range(0, length, batch_size):
            idx = np.arange(start, min(start + batch_size, length))
            pairs = np.stack((seq[idx], seq[idx + 1]), axis=1)
            all_logits.append(model(torch.from_numpy(pairs).to(device=device, dtype=torch.float16)).float().cpu())
            all_labels.append(torch.from_numpy(labels[sum(lengths[:0]):]))
            # Labels are consumed globally below; this placeholder is replaced by
            # the explicit global cursor in the outer implementation.
        # unreachable: kept for structure
    raise AssertionError("student evaluator replaced at runtime")


@torch.inference_mode()
def evaluate_student(model, embeddings: np.memmap, labels: np.ndarray, valid: np.ndarray, lengths: np.ndarray, device: torch.device, batch_size: int) -> dict:
    tp = fp = tn = fn = count = 0
    loss_sum = 0.0
    score_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    sentence_cursor = label_cursor = 0
    for length in lengths:
        if length <= 0:
            continue
        sentence_count = int(length)
        seq = np.asarray(embeddings[sentence_cursor:sentence_cursor + sentence_count], dtype=np.float32)
        sentence_cursor += sentence_count
        pair_count = int(length) - 1
        for start in range(0, pair_count, batch_size):
            end = min(start + batch_size, pair_count)
            pairs = np.stack((seq[start:end], seq[start + 1:end + 1]), axis=1)
            mask = valid[label_cursor + start:label_cursor + end].astype(bool)
            pairs = pairs[mask]
            if not len(pairs):
                continue
            target = torch.from_numpy(labels[label_cursor + start:label_cursor + end][mask]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(torch.from_numpy(pairs).to(device=device, dtype=torch.float16))
                loss = F.cross_entropy(logits.float(), target)
            score_rows.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            label_rows.append(target.cpu().numpy())
            pred = logits.argmax(-1)
            tp += int(((pred == 1) & (target == 1)).sum())
            fp += int(((pred == 1) & (target == 0)).sum())
            tn += int(((pred == 0) & (target == 0)).sum())
            fn += int(((pred == 0) & (target == 1)).sum())
            count += target.numel()
            loss_sum += float(loss) * target.numel()
        label_cursor += int(length)
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    result = {"examples": count, "loss": loss_sum / count, "accuracy": (tp + tn) / count,
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    result["score_diagnostics"] = score_diagnostics(np.concatenate(score_rows), np.concatenate(label_rows))
    return result


@torch.inference_mode()
def evaluate_baseline(model, tokenizer, texts: list[str], labels: np.ndarray, valid: np.ndarray, lengths: np.ndarray, device: torch.device, batch_size: int) -> dict:
    tp = fp = tn = fn = count = 0; loss_sum = 0.0; label_cursor = 0
    score_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for doc_index, text in enumerate(texts):
        sentences = text.split("\n")
        if sentences and sentences[-1] == "":
            sentences.pop()
        length = int(lengths[doc_index])
        if len(sentences) != length:
            raise RuntimeError(f"sentence/label mismatch at document {doc_index}: {len(sentences)} != {length}")
        # The model-card executable examples and widget use Left [SEP] Right.
        pairs = [(sentences[i], sentences[i + 1]) for i in range(length - 1)]
        for start in range(0, length, batch_size):
            batch_pairs = pairs[start:start + batch_size]
            valid_mask = valid[label_cursor + start:label_cursor + start + len(batch_pairs)].astype(bool)
            batch_pairs = [pair for pair, keep in zip(batch_pairs, valid_mask) if keep]
            if not batch_pairs:
                continue
            encoded = tokenizer([a for a, _ in batch_pairs], [b for _, b in batch_pairs], padding=True, truncation=True, max_length=512, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            target = torch.from_numpy(labels[label_cursor + start:label_cursor + start + len(valid_mask)][valid_mask]).to(device)
            logits = model(**encoded).logits
            score_rows.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            label_rows.append(target.cpu().numpy())
            loss_sum += float(F.cross_entropy(logits.float(), target)) * target.numel()
            pred = logits.argmax(-1)
            tp += int(((pred == 1) & (target == 1)).sum()); fp += int(((pred == 1) & (target == 0)).sum())
            tn += int(((pred == 0) & (target == 0)).sum()); fn += int(((pred == 0) & (target == 1)).sum()); count += target.numel()
        label_cursor += length
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    result = {"examples": count, "loss": loss_sum / count, "accuracy": (tp + tn) / count,
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    result["score_diagnostics"] = score_diagnostics(np.concatenate(score_rows), np.concatenate(label_rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--only", choices=["student", "baseline", "both"], default="both")
    parser.add_argument("--max-documents", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    texts, lengths = load_rows(args.parquet)
    if args.max_documents:
        texts = texts[:args.max_documents]
        lengths = lengths[:args.max_documents]
    labels = np.fromfile(args.labels, dtype=np.uint8).astype(np.int64)
    valid = np.fromfile(args.valid, dtype=np.uint8)
    embeddings = np.memmap(args.embeddings, dtype=np.float16, mode="r", shape=(int(embeddings_size := args.embeddings.stat().st_size // (2 * 768)), 768))
    expected_rows = int(lengths.sum())
    if len(labels) != embeddings.shape[0] or len(valid) != embeddings.shape[0] or expected_rows > embeddings.shape[0]:
        raise RuntimeError(f"cache/data mismatch: embeddings={embeddings.shape} labels={len(labels)} docs={len(lengths)} lengths={lengths.sum()}")
    started = time.time()
    result = {"dataset": str(args.parquet), "documents": len(texts), "pairs": int(valid[:expected_rows].sum()), "device": torch.cuda.get_device_name(0)}
    if args.only in ("student", "both"):
        backbone = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").to(device).eval()
        student = ContinuousPairClassifier(backbone)
        state = load_file(str(args.student), device="cpu")
        student.load_state_dict(state, strict=False); student.to(device).eval()
        result["student"] = evaluate_student(student, embeddings, labels, valid, lengths, device, args.batch_size)
        del student, backbone, state
        torch.cuda.empty_cache()
    if args.only in ("baseline", "both"):
        baseline_name = "BlueOrangeDigital/distilbert-cross-segment-document-chunking"
        tokenizer = AutoTokenizer.from_pretrained(baseline_name)
        baseline = AutoModelForSequenceClassification.from_pretrained(baseline_name).to(device).eval()
        result["baseline"] = evaluate_baseline(baseline, tokenizer, texts, labels, valid, lengths, device, args.batch_size)
    result["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
