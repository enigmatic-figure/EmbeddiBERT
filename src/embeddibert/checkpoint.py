from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def apply_first_layer_husk(bert_model, checkpoint: str | Path):
    """Apply an EmbeddiBERT first-layer husk to a compatible BertModel in place."""
    state = load_file(str(checkpoint), device="cpu")
    word_embeddings = bert_model.embeddings.word_embeddings.weight
    replacement = state.pop("word_embeddings")
    if replacement.shape != word_embeddings.shape:
        raise ValueError(
            f"Checkpoint word table {replacement.shape} does not match model "
            f"table {word_embeddings.shape}"
        )
    with torch.no_grad():
        word_embeddings.copy_(replacement.to(word_embeddings.dtype))

    layer = bert_model.encoder.layer[0]
    for prefix, module in (
        ("attention.", layer.attention),
        ("intermediate.", layer.intermediate),
        ("output.", layer.output),
    ):
        module_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        module.load_state_dict(module_state, strict=True)
    return bert_model

