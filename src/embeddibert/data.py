from __future__ import annotations

import random

import torch


CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A transformer maps a sequence into contextual representations.",
    "The compiler rejected the program because the variable was undefined.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;",
    "Scientific models should be tested against falsifiable measurements.",
    "The package includes a tokenizer, configuration, and learned parameters.",
    "Attention compares queries with keys and aggregates the corresponding values.",
    "Small controlled experiments make architectural failures easier to diagnose.",
    "Please summarize the document and identify its strongest counterargument.",
    "An embedding table assigns one vector to every discrete token identifier.",
    "Machine learning systems can fail silently when padding masks are incorrect.",
    "Yesterday the weather was cold, but tomorrow may be warm and clear.",
    "Users expect reliable software, useful errors, and reproducible results.",
    "Rust prevents many memory errors while Python favors rapid experimentation.",
    "A database transaction should either finish completely or roll back safely.",
]


def make_calibration_batch(
    tokenizer,
    *,
    batch_size: int,
    sequence_length: int,
    generator: random.Random,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    for _ in range(batch_size):
        count = generator.randint(1, 4)
        parts = generator.choices(CALIBRATION_TEXTS, k=count)
        texts.append(" ".join(parts))
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}

