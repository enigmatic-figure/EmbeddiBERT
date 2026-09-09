import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from embeddibert.mlm_alignment import (
    MlmAlignmentConfig,
    configure_trainable_parameters,
    distillation_metrics,
    make_masked_batch,
    prediction_logits,
    train_stage,
)


def tiny_model() -> DistilBertForMaskedLM:
    config = DistilBertConfig(
        vocab_size=31,
        max_position_embeddings=32,
        dim=16,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = DistilBertForMaskedLM(config)
    model.tie_weights()
    return model


def test_masked_batch_changes_only_eligible_positions() -> None:
    corpus = {
        "input_ids": torch.tensor([[1, 5, 6, 2, 0], [1, 7, 8, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
        "vocab_size": torch.tensor(31),
    }
    corrupted, attention_mask, selected = make_masked_batch(
        corpus,
        batch_size=2,
        generator=torch.Generator().manual_seed(3),
        mask_token_id=4,
        special_token_ids={0, 1, 2, 4},
        mask_probability=1.0,
    )

    assert torch.equal(attention_mask, corpus["attention_mask"])
    assert selected.tolist() == [
        [False, True, True, False, False],
        [False, True, True, False, False],
    ]
    assert torch.equal(corrupted[~selected], corpus["input_ids"][~selected])


def test_interpretation_stage_freezes_attention_and_ffn() -> None:
    model = tiny_model()
    manifest = configure_trainable_parameters(model, "interpretation")

    trainable = set(manifest["trainable_names"])
    assert "vocab_transform.weight" in trainable
    assert "vocab_layer_norm.weight" in trainable
    assert "vocab_projector.bias" in trainable
    assert "distilbert.embeddings.position_embeddings.weight" in trainable
    assert any("sa_layer_norm.weight" in name for name in trainable)
    assert not any(".attention." in name for name in trainable)
    assert not any(".ffn.lin" in name for name in trainable)
    assert not model.distilbert.embeddings.word_embeddings.weight.requires_grad


def test_whole_model_stage_unlocks_encoder_but_keeps_qwen_table_fixed() -> None:
    model = tiny_model()
    manifest = configure_trainable_parameters(model, "whole_model")

    trainable = set(manifest["trainable_names"])
    assert any(".attention.q_lin.weight" in name for name in trainable)
    assert any(".ffn.lin1.weight" in name for name in trainable)
    assert "vocab_transform.weight" in trainable
    assert not model.distilbert.embeddings.word_embeddings.weight.requires_grad
    assert not model.vocab_projector.weight.requires_grad


def test_masked_prediction_and_distillation_are_finite() -> None:
    teacher = tiny_model().eval()
    student = tiny_model().eval()
    ids = torch.tensor([[1, 4, 9, 2], [1, 10, 4, 2]])
    mask = torch.ones_like(ids)
    selected = ids.eq(4)

    with torch.no_grad():
        teacher_logits = prediction_logits(teacher, ids, mask, selected)
        student_logits = prediction_logits(student, ids, mask, selected)
    loss, metrics = distillation_metrics(student_logits, teacher_logits, 2.0)

    assert teacher_logits.shape == (2, 31)
    assert torch.isfinite(loss)
    assert metrics["teacher_kl"] >= 0.0
    assert 0.0 <= metrics["top1_agreement"] <= 1.0


def test_interpretation_training_step_runs_end_to_end() -> None:
    teacher = tiny_model().eval()
    student = tiny_model().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    corpus = {
        "input_ids": torch.tensor(
            [[1, 5, 6, 7, 2, 0], [1, 8, 9, 10, 2, 0], [1, 11, 12, 13, 2, 0]]
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]]
        ),
        "vocab_size": torch.tensor(31),
    }
    config = MlmAlignmentConfig(
        sequence_length=6,
        train_batch_size=2,
        eval_batch_size=2,
        eval_batches=1,
        interpretation_max_steps=1,
        whole_model_max_steps=1,
        eval_every=1,
        early_stopping_patience=1,
        use_amp=False,
    )

    result = train_stage(
        teacher,
        student,
        corpus,
        corpus,
        stage="interpretation",
        max_steps=1,
        learning_rate=1e-4,
        config=config,
        mask_token_id=4,
        special_token_ids={0, 1, 2, 4},
        teacher_device=torch.device("cpu"),
        student_device=torch.device("cpu"),
        log=lambda _: None,
    )

    assert result["evaluations"][-1]["teacher_kl"] >= 0.0
    assert result["manifest"]["trainable_parameters"] > 0
