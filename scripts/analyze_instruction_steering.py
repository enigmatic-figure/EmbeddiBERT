#!/usr/bin/env python3
"""Analyze one document-disjoint slice of an instruction-steering round.

The evaluator deliberately writes raw per-pair scores so prompt development can
be restricted to a declared document subset.  This script keeps that split
explicit and estimates uncertainty by resampling whole documents, not pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_instruction_steering import classification_metrics, score_shift

BOOTSTRAP_METRICS = (
    "roc_auc",
    "average_precision",
    "best_slice_f1",
    "f1",
    "balanced_accuracy",
    "brier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", required=True, type=Path)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--document-start", required=True, type=int)
    parser.add_argument("--document-stop", required=True, type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p025": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.5)),
        "p975": float(np.quantile(values, 0.975)),
        "probability_above_zero": float(np.mean(values > 0)),
    }


def bootstrap_indices(
    document_ids: np.ndarray, repetitions: int, seed: int
) -> list[np.ndarray]:
    documents = np.unique(document_ids)
    grouped = {document: np.flatnonzero(document_ids == document) for document in documents}
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for _ in range(repetitions):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        rows.append(np.concatenate([grouped[document] for document in sampled]))
    return rows


def paired_bootstrap(
    labels: np.ndarray,
    anchor: np.ndarray,
    contender: np.ndarray,
    resamples: list[np.ndarray],
) -> dict[str, dict[str, float]]:
    differences = {metric: [] for metric in BOOTSTRAP_METRICS}
    for indices in resamples:
        anchor_metrics = classification_metrics(labels[indices], anchor[indices])
        contender_metrics = classification_metrics(labels[indices], contender[indices])
        for metric in BOOTSTRAP_METRICS:
            differences[metric].append(contender_metrics[metric] - anchor_metrics[metric])
    return {
        metric: interval(np.asarray(values, dtype=np.float64))
        for metric, values in differences.items()
    }


def main() -> None:
    args = parse_args()
    if args.document_start < 0 or args.document_stop <= args.document_start:
        raise ValueError("document range must be a non-empty half-open interval")
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")

    manifest = json.loads(args.conditions.read_text(encoding="utf-8"))
    condition_ids = [row["id"] for row in manifest["conditions"]]
    anchor_id = manifest["anchor_id"]
    if anchor_id not in condition_ids:
        raise ValueError("anchor_id is not present in the condition manifest")

    labels_all = np.load(args.round_dir / "labels.npy")
    document_ids_all = np.load(args.round_dir / "pair_document_ids.npy")
    selected = (document_ids_all >= args.document_start) & (
        document_ids_all < args.document_stop
    )
    labels = labels_all[selected]
    document_ids = document_ids_all[selected]
    if not selected.any():
        raise ValueError("document range selects no pairs")
    observed_documents = np.unique(document_ids)
    expected_documents = np.arange(args.document_start, args.document_stop)
    if not np.array_equal(observed_documents, expected_documents):
        raise ValueError("document range is incomplete; refusing a silently changed slice")

    scores: dict[str, np.ndarray] = {}
    for condition_id in condition_ids:
        path = args.round_dir / f"scores.{condition_id}.npy"
        values = np.load(path)
        if values.shape != labels_all.shape:
            raise ValueError(f"score shape mismatch for {condition_id}: {values.shape}")
        scores[condition_id] = values[selected]

    anchor = scores[anchor_id]
    resamples = bootstrap_indices(
        document_ids, args.bootstrap_repetitions, args.seed
    )
    analyses: dict[str, Any] = {}
    for condition_id in condition_ids:
        contender = scores[condition_id]
        analyses[condition_id] = {
            "metrics": classification_metrics(labels, contender),
            "versus_anchor": score_shift(anchor, contender),
            "paired_document_bootstrap_delta_versus_anchor": paired_bootstrap(
                labels, anchor, contender, resamples
            ),
        }

    result = {
        "round_id": manifest["round_id"],
        "anchor_id": anchor_id,
        "scope": {
            "document_start_inclusive": args.document_start,
            "document_stop_exclusive": args.document_stop,
            "documents": len(observed_documents),
            "pairs": len(labels),
            "boundaries": int(labels.sum()),
            "positive_rate": float(labels.mean()),
        },
        "bootstrap": {
            "unit": "document",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "interval": "percentile_95",
            "delta_direction": "condition_minus_anchor; lower is better only for brier",
        },
        "conditions": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
