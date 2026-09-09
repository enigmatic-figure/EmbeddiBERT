import torch

from transformers import BertConfig, BertModel

from embeddibert.alignment import attention_kl, make_first_layer_modules, masked_mse
from embeddibert.embedding_table import last_token_pool, render_wordpiece


def test_surface_rendering_removes_wordpiece_marker():
    assert render_wordpiece("##ing", "surface") == "ing"
    assert render_wordpiece("hello", "surface") == "hello"
    assert render_wordpiece("[CLS]", "surface") == "[CLS]"


def test_last_token_pool_supports_right_padding():
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    pooled = last_token_pool(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_last_token_pool_supports_left_padding():
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    pooled = last_token_pool(hidden, mask)
    assert torch.equal(pooled, hidden[:, -1])


def test_masked_losses_ignore_padding_and_match_identity():
    tensor = torch.randn(2, 3, 5)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert masked_mse(tensor, tensor, mask).item() == 0.0
    probs = torch.softmax(torch.randn(2, 4, 3, 3), dim=-1)
    assert abs(attention_kl(probs, probs, mask).item()) < 1e-6


def test_first_layer_modules_accept_exact_replacement_table():
    config = BertConfig(
        vocab_size=17,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=24,
    )
    bert = BertModel(config)
    replacement = torch.randn(17, 12)
    modules = make_first_layer_modules(bert, replacement)
    assert torch.equal(modules.student_embeddings.word_embeddings.weight, replacement)
