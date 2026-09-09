from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Iterator


def split_document(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def deterministic_span(
    text: str,
    *,
    document_index: int,
    seed: int,
    min_sentences: int = 2,
    max_sentences: int = 8,
) -> str | None:
    sentences = split_document(text)
    if len(sentences) < min_sentences:
        return None
    digest = hashlib.sha256(f"{seed}:{document_index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    width = rng.randint(min_sentences, min(max_sentences, len(sentences)))
    start = rng.randint(0, len(sentences) - width)
    return " ".join(sentences[start : start + width])


def deterministic_partition(document_index: int, seed: int, evaluation_percent: int) -> str:
    digest = hashlib.sha256(f"partition:{seed}:{document_index}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "evaluation" if bucket < evaluation_percent else "train"


def load_alignment_jsonl(path: str | Path) -> tuple[list[str], list[str]]:
    train: list[str] = []
    evaluation: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            destination = evaluation if row["partition"] == "evaluation" else train
            destination.append(row["text"])
    if not train or not evaluation:
        raise ValueError("Alignment corpus must contain train and evaluation partitions")
    return train, evaluation


def adjacent_pairs(text: str, labels: Iterable[int]) -> Iterator[tuple[str, str, int]]:
    sentences = split_document(text)
    label_list = list(labels)
    if len(sentences) != len(label_list):
        raise ValueError(
            f"Sentence/label length mismatch: {len(sentences)} != {len(label_list)}"
        )
    for index in range(len(sentences) - 1):
        yield sentences[index], sentences[index + 1], int(label_list[index])
