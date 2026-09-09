from pathlib import Path

import pytest

from kaggle_distil_entry import resolve_corpus_path


def test_resolve_corpus_path_keeps_existing_file(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    payload = {"corpus_path": str(corpus)}

    resolve_corpus_path(payload, tmp_path / "unused")

    assert payload["corpus_path"] == str(corpus)


def test_resolve_corpus_path_finds_rewritten_kaggle_mount(tmp_path: Path) -> None:
    corpus = tmp_path / "owner-dataset-v2" / "corpus.jsonl"
    corpus.parent.mkdir()
    corpus.write_text("{}\n", encoding="utf-8")
    payload = {"corpus_path": "/kaggle/input/original-slug/corpus.jsonl"}

    resolve_corpus_path(payload, tmp_path)

    assert payload["corpus_path"] == str(corpus)


def test_resolve_corpus_path_rejects_missing_input(tmp_path: Path) -> None:
    payload = {"corpus_path": "/kaggle/input/original-slug/corpus.jsonl"}

    with pytest.raises(FileNotFoundError, match="Could not resolve"):
        resolve_corpus_path(payload, tmp_path)
