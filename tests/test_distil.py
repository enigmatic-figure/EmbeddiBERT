import copy

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel

from embeddibert.cross_segment import QwenPairClassifier
from embeddibert.distil_alignment import (
    DistilAlignmentConfig,
    evaluate_composition,
    evaluate_isolated_layer,
    train_independent_layer,
)


def tiny_config(num_layers=2):
    config = DistilBertConfig(
        vocab_size=17,
        max_position_embeddings=32,
        dim=12,
        hidden_dim=24,
        n_layers=num_layers,
        n_heads=3,
        dropout=0.0,
        attention_dropout=0.0,
        num_labels=2,
    )
    config._attn_implementation = "eager"
    return config


def test_independent_and_composed_distil_metrics_are_finite():
    teacher = DistilBertModel(tiny_config())
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.embeddings.word_embeddings.weight.copy_(torch.randn(17, 12))
    corpus = {
        "input_ids": torch.randint(0, 17, (8, 10)),
        "attention_mask": torch.ones(8, 10, dtype=torch.long),
    }
    isolated = evaluate_isolated_layer(
        teacher,
        student,
        0,
        corpus,
        batches=2,
        batch_size=2,
        seed=1,
        device=torch.device("cpu"),
    )
    composed = evaluate_composition(
        teacher,
        student,
        corpus,
        batches=2,
        batch_size=2,
        seed=1,
        device=torch.device("cpu"),
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in isolated.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in composed.values())


def test_qwen_pair_classifier_injects_two_continuous_inputs():
    base = DistilBertForSequenceClassification(tiny_config(num_layers=1))
    model = QwenPairClassifier(base, cls_token_id=1, sep_token_id=2)
    logits = model(torch.randn(3, 12), torch.randn(3, 12))
    assert logits.shape == (3, 2)


def test_all_independent_training_stages_execute():
    teacher = DistilBertModel(tiny_config(num_layers=1)).eval()
    student = copy.deepcopy(teacher).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        student.embeddings.word_embeddings.weight.copy_(torch.randn(17, 12))
    corpus = {
        "input_ids": torch.randint(0, 17, (8, 10)),
        "attention_mask": torch.ones(8, 10, dtype=torch.long),
    }
    config = DistilAlignmentConfig(
        train_batch_size=2,
        attention_steps_per_layer=1,
        intermediate_steps_per_layer=1,
        output_steps_per_layer=1,
        log_every=1,
    )
    records = []
    train_independent_layer(
        teacher,
        student,
        0,
        corpus,
        config=config,
        device=torch.device("cpu"),
        log=records.append,
    )
    assert [record["stage"] for record in records] == [
        "attention",
        "intermediate",
        "output",
    ]
