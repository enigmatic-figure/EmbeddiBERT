import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "evaluate_instruction_steering", SCRIPTS / "evaluate_instruction_steering.py"
)
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
assert EVALUATOR_SPEC.loader is not None
sys.modules[EVALUATOR_SPEC.name] = EVALUATOR
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)

ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_instruction_steering", SCRIPTS / "analyze_instruction_steering.py"
)
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
assert ANALYSIS_SPEC.loader is not None
sys.modules[ANALYSIS_SPEC.name] = ANALYSIS
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def input_artifacts(round_dir: Path, manifest: Path) -> dict[str, object]:
    return {
        "analysis_contract": ANALYSIS.ANALYSIS_CONTRACT,
        "conditions_manifest_sha256": ANALYSIS.file_sha256(manifest),
        "labels_sha256": ANALYSIS.file_sha256(round_dir / "labels.npy"),
        "pair_document_ids_sha256": ANALYSIS.file_sha256(
            round_dir / "pair_document_ids.npy"
        ),
        "score_sha256": {
            "anchor": ANALYSIS.file_sha256(round_dir / "scores.anchor.npy")
        },
    }


def test_metrics_at_threshold_are_exact() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    scores = np.asarray([0.1, 0.8, 0.6, 0.9], dtype=np.float32)
    metrics = ANALYSIS.metrics_at_threshold(labels, scores, 0.7)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["f1"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5


def test_analysis_transfers_threshold_from_disjoint_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    np.save(round_dir / "labels.npy", np.asarray([0, 1, 0, 1], dtype=np.uint8))
    np.save(
        round_dir / "pair_document_ids.npy",
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    np.save(
        round_dir / "scores.anchor.npy",
        np.asarray([0.1, 0.8, 0.4, 0.9], dtype=np.float32),
    )
    manifest = tmp_path / "conditions.json"
    manifest.write_text(
        json.dumps(
            {
                "round_id": "test-round",
                "anchor_id": "anchor",
                "conditions": [{"id": "anchor", "instruction": "test"}],
            }
        ),
        encoding="utf-8",
    )
    threshold_source = tmp_path / "thresholds.json"
    threshold_source.write_text(
        json.dumps(
            {
                "round_id": "test-round",
                "input_artifacts": input_artifacts(round_dir, manifest),
                "scope": {
                    "document_start_inclusive": 2,
                    "document_stop_exclusive": 4,
                },
                "conditions": {
                    "anchor": {"metrics": {"best_slice_threshold": 0.7}}
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_instruction_steering.py",
            "--round-dir",
            str(round_dir),
            "--conditions",
            str(manifest),
            "--document-start",
            "0",
            "--document-stop",
            "2",
            "--bootstrap-repetitions",
            "3",
            "--threshold-source",
            str(threshold_source),
            "--output",
            str(output),
        ],
    )
    ANALYSIS.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    transferred = result["conditions"]["anchor"]["transferred_tuning_threshold"]
    assert transferred["metrics"]["threshold"] == 0.7
    assert transferred["metrics"]["f1"] == 1.0
    assert result["threshold_source"]["document_start_inclusive"] == 2


def test_analysis_rejects_overlapping_threshold_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    np.save(round_dir / "labels.npy", np.asarray([0, 1], dtype=np.uint8))
    np.save(
        round_dir / "pair_document_ids.npy", np.asarray([0, 0], dtype=np.int32)
    )
    np.save(
        round_dir / "scores.anchor.npy", np.asarray([0.1, 0.9], dtype=np.float32)
    )
    manifest = tmp_path / "conditions.json"
    manifest.write_text(
        json.dumps(
            {
                "round_id": "test-round",
                "anchor_id": "anchor",
                "conditions": [{"id": "anchor", "instruction": "test"}],
            }
        ),
        encoding="utf-8",
    )
    threshold_source = tmp_path / "thresholds.json"
    threshold_source.write_text(
        json.dumps(
            {
                "round_id": "test-round",
                "input_artifacts": input_artifacts(round_dir, manifest),
                "scope": {
                    "document_start_inclusive": 0,
                    "document_stop_exclusive": 1,
                },
                "conditions": {
                    "anchor": {"metrics": {"best_slice_threshold": 0.5}}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_instruction_steering.py",
            "--round-dir",
            str(round_dir),
            "--conditions",
            str(manifest),
            "--document-start",
            "0",
            "--document-stop",
            "1",
            "--bootstrap-repetitions",
            "1",
            "--threshold-source",
            str(threshold_source),
            "--output",
            str(tmp_path / "analysis.json"),
        ],
    )
    with pytest.raises(ValueError, match="overlaps"):
        ANALYSIS.main()


def test_analysis_rejects_thresholds_from_changed_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    np.save(round_dir / "labels.npy", np.asarray([0, 1], dtype=np.uint8))
    np.save(
        round_dir / "pair_document_ids.npy", np.asarray([0, 0], dtype=np.int32)
    )
    score_path = round_dir / "scores.anchor.npy"
    np.save(score_path, np.asarray([0.1, 0.9], dtype=np.float32))
    manifest = tmp_path / "conditions.json"
    manifest.write_text(
        json.dumps(
            {
                "round_id": "test-round",
                "anchor_id": "anchor",
                "conditions": [{"id": "anchor", "instruction": "test"}],
            }
        ),
        encoding="utf-8",
    )
    threshold_source = tmp_path / "thresholds.json"
    threshold_source.write_text(
        json.dumps(
            {
                "round_id": "test-round",
                "input_artifacts": input_artifacts(round_dir, manifest),
                "scope": {
                    "document_start_inclusive": 1,
                    "document_stop_exclusive": 2,
                },
                "conditions": {
                    "anchor": {"metrics": {"best_slice_threshold": 0.5}}
                },
            }
        ),
        encoding="utf-8",
    )
    np.save(score_path, np.asarray([0.2, 0.8], dtype=np.float32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_instruction_steering.py",
            "--round-dir",
            str(round_dir),
            "--conditions",
            str(manifest),
            "--document-start",
            "0",
            "--document-stop",
            "1",
            "--bootstrap-repetitions",
            "1",
            "--threshold-source",
            str(threshold_source),
            "--output",
            str(tmp_path / "analysis.json"),
        ],
    )
    with pytest.raises(ValueError, match="exact manifest"):
        ANALYSIS.main()
