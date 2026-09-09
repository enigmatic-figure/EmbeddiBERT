from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def apply_distilbert_husk(
    model: torch.nn.Module,
    husk_path: str | Path,
    *,
    load_word_embeddings: bool = True,
) -> torch.nn.Module:
    """Apply an EmbeddiBERT DistilBERT husk to a compatible base model."""
    state = load_file(str(husk_path), device="cpu")
    expected = {"word_embeddings"}
    expected.update(
        f"layer.{index}.{key}"
        for index, layer in enumerate(model.transformer.layer)
        for key in layer.state_dict()
    )
    missing = expected.difference(state)
    extra = set(state).difference(expected)
    if missing or extra:
        raise ValueError(
            f"Incompatible husk: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    if load_word_embeddings:
        destination = model.embeddings.word_embeddings.weight
        source = state["word_embeddings"]
        if source.shape != destination.shape:
            raise ValueError(
                f"Word embedding shape {tuple(source.shape)} != {tuple(destination.shape)}"
            )
        with torch.no_grad():
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))

    for index, layer in enumerate(model.transformer.layer):
        prefix = f"layer.{index}."
        layer_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        layer.load_state_dict(layer_state, strict=True)
    return model
