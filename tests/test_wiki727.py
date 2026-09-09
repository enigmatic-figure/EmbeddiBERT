import pytest

from embeddibert.wiki727 import (
    adjacent_pairs,
    deterministic_partition,
    deterministic_span,
)


def test_adjacent_pair_label_is_boundary_after_left_sentence():
    pairs = list(adjacent_pairs("one\ntwo\nthree", [0, 1, 1]))
    assert pairs == [("one", "two", 0), ("two", "three", 1)]


def test_adjacent_pair_rejects_misaligned_labels():
    with pytest.raises(ValueError):
        list(adjacent_pairs("one\ntwo", [0]))


def test_deterministic_span_is_stable_and_contiguous():
    text = "\n".join(f"sentence {index}" for index in range(12))
    first = deterministic_span(text, document_index=42, seed=7)
    second = deterministic_span(text, document_index=42, seed=7)
    assert first == second
    assert 2 <= first.count("sentence") <= 8


def test_deterministic_partition_is_stable():
    assert deterministic_partition(42, 7, 10) == deterministic_partition(42, 7, 10)
