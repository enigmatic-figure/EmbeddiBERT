import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import DistilBertConfig, DistilBertModel


SCRIPT = Path(__file__).parents[1] / "scripts" / "colab_train_continuous_pairs.py"
SPEC = importlib.util.spec_from_file_location("continuous_training_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_split_exact_preserves_internal_empty_sentences() -> None:
    assert MODULE.split_exact("first\n\nthird\n", 3) == ["first", "", "third"]
    with pytest.raises(ValueError, match="Sentence/label mismatch"):
        MODULE.split_exact("first\nsecond\n", 3)


def test_continuous_classifier_has_no_word_table_and_accepts_two_vectors() -> None:
    config = DistilBertConfig(
        vocab_size=11,
        max_position_embeddings=4,
        dim=768,
        hidden_dim=32,
        n_layers=1,
        n_heads=12,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
    )
    backbone = DistilBertModel(config)
    del backbone.embeddings.word_embeddings
    model = MODULE.ContinuousPairClassifier(backbone).eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 2, 768))

    assert logits.shape == (2, 2)
    assert not any("word_embeddings" in name for name, _ in model.named_parameters())


def test_continuous_classifier_rejects_token_shaped_inputs() -> None:
    config = DistilBertConfig(
        vocab_size=11,
        max_position_embeddings=4,
        dim=768,
        hidden_dim=32,
        n_layers=1,
        n_heads=12,
    )
    backbone = DistilBertModel(config)
    del backbone.embeddings.word_embeddings
    model = MODULE.ContinuousPairClassifier(backbone)

    with pytest.raises(ValueError, match=r"\[batch, 2, 768\]"):
        model(torch.empty(2, 768))
