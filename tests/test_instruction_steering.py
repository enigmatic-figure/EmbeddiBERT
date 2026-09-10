import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_instruction_steering.py"
SPEC = importlib.util.spec_from_file_location("instruction_steering", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_condition_manifest_supports_symmetric_and_role_specific_prompts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conditions.json"
    path.write_text(
        json.dumps(
            {
                "round_id": "round-test",
                "anchor_id": "anchor",
                "conditions": [
                    {"id": "anchor", "instruction": "same"},
                    {
                        "id": "roles",
                        "left_instruction": "left",
                        "right_instruction": "right",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    round_id, anchor_id, conditions, _ = MODULE.load_conditions(path)
    assert round_id == "round-test"
    assert anchor_id == "anchor"
    assert conditions[0].left_instruction == conditions[0].right_instruction == "same"
    assert conditions[1].left_instruction == "left"
    assert conditions[1].right_instruction == "right"


def test_condition_manifest_rejects_missing_anchor(tmp_path: Path) -> None:
    path = tmp_path / "conditions.json"
    path.write_text(
        json.dumps(
            {
                "round_id": "round-test",
                "anchor_id": "missing",
                "conditions": [{"id": "present", "instruction": ""}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent"):
        MODULE.load_conditions(path)


def test_document_sample_preserves_document_boundaries_and_left_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dev.parquet"
    table = pa.table(
        {
            "text": ["one\ntwo\nthree", "alpha\nbeta"],
            "label": [[0, 1, 1], [1, 1]],
        }
    )
    pq.write_table(table, path)
    sample = MODULE.load_document_sample(path, 2)
    assert sample.sentences == ["one", "two", "three", "alpha", "beta"]
    assert sample.labels.tolist() == [0, 1, 1]
    assert sample.pair_document_ids.tolist() == [0, 0, 1]
    assert sample.sentence_offsets.tolist() == [0, 3, 5]
    assert sample.pair_offsets.tolist() == [0, 2, 3]
    with pytest.raises(ValueError, match="sample contract mismatch"):
        MODULE.validate_canonical_sample(sample)


def test_canonical_artifact_contract_fails_closed() -> None:
    MODULE.validate_canonical_artifacts(
        max_documents=MODULE.EXPECTED_DOCUMENTS,
        parquet_sha256=MODULE.EXPECTED_DEV_PARQUET_SHA256,
        student_sha256=MODULE.EXPECTED_STUDENT_SHA256,
        source_commit="a" * 40,
    )
    with pytest.raises(ValueError, match="pinned Wiki-727K dev parquet"):
        MODULE.validate_canonical_artifacts(
            max_documents=MODULE.EXPECTED_DOCUMENTS,
            parquet_sha256="wrong",
            student_sha256=MODULE.EXPECTED_STUDENT_SHA256,
            source_commit="a" * 40,
        )


def test_metrics_and_anchor_shift_are_exact() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    anchor = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    shifted = np.asarray([0.2, 0.7, 0.4, 0.8], dtype=np.float32)
    metrics = MODULE.classification_metrics(labels, anchor)
    shift = MODULE.score_shift(anchor, shifted)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert shift["decision_flip_rate"] == 0.5
    assert shift["negative_to_positive"] == 1
    assert shift["positive_to_negative"] == 1
    assert shift["mean_delta"] == pytest.approx(0.025)


def test_rank_metrics_group_ties_and_are_permutation_invariant() -> None:
    labels = np.asarray([1, 0], dtype=np.uint8)
    scores = np.asarray([0.5, 0.5], dtype=np.float32)
    forward = MODULE.rank_metrics(labels, scores)
    reverse = MODULE.rank_metrics(labels[::-1], scores[::-1])
    assert forward == reverse
    assert forward["roc_auc"] == 0.5
    assert forward["average_precision"] == 0.5
    assert forward["best_slice_f1"] == pytest.approx(2 / 3)
    assert forward["best_slice_threshold"] == 0.5


def test_cosine_summary_renormalizes_float16_rows(tmp_path: Path) -> None:
    first_path = tmp_path / "first.f16"
    second_path = tmp_path / "second.f16"
    first = np.memmap(first_path, dtype=np.float16, mode="w+", shape=(2, 3))
    second = np.memmap(
        second_path, dtype=np.float16, mode="w+", shape=(2, 3)
    )
    first[:] = [[2, 0, 0], [0, 3, 0]]
    second[:] = [[5, 0, 0], [0, -7, 0]]
    first.flush()
    second.flush()
    summary = MODULE.cosine_summary(first, second)
    assert summary["mean"] == 0.0
    assert summary["p05"] == pytest.approx(-0.9)
    assert summary["p95"] == pytest.approx(0.9)


def test_cache_identity_binds_instruction_and_sample() -> None:
    first = MODULE.cache_identity("sample-a", "instruction-a", 32768)
    assert first == MODULE.cache_identity("sample-a", "instruction-a", 32768)
    assert first != MODULE.cache_identity("sample-a", "instruction-b", 32768)
    assert first != MODULE.cache_identity("sample-b", "instruction-a", 32768)


def test_pair_batches_never_cross_documents(tmp_path: Path) -> None:
    path = tmp_path / "vectors.f16"
    vectors = np.memmap(path, dtype=np.float16, mode="w+", shape=(5, MODULE.DIMENSION))
    vectors[:] = 0
    vectors[:, 0] = np.arange(5)
    vectors.flush()
    sample = MODULE.DocumentSample(
        sentences=["one", "two", "three", "alpha", "beta"],
        labels=np.asarray([0, 1, 1], dtype=np.uint8),
        pair_document_ids=np.asarray([0, 0, 1], dtype=np.int32),
        sentence_offsets=np.asarray([0, 3, 5], dtype=np.int64),
        pair_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        document_indices=np.asarray([0, 1], dtype=np.int64),
        digest="sample",
    )
    batches = list(MODULE.iter_pair_batches(vectors, vectors, sample, batch_size=8))
    assert [offset for offset, _ in batches] == [0, 2]
    assert batches[0][1][:, :, 0].tolist() == [[0, 1], [1, 2]]
    assert batches[1][1][:, :, 0].tolist() == [[3, 4]]
