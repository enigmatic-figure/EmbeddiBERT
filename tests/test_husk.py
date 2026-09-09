from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from embeddibert.husk import apply_distilbert_husk


class TinyDistilBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Module()
        self.embeddings.word_embeddings = torch.nn.Embedding(7, 3)
        self.transformer = torch.nn.Module()
        self.transformer.layer = torch.nn.ModuleList(
            [torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)]
        )


def make_husk(model: TinyDistilBert, path: Path) -> dict[str, torch.Tensor]:
    state = {"word_embeddings": torch.full((7, 3), 0.25)}
    for index, layer in enumerate(model.transformer.layer):
        state.update(
            {
                f"layer.{index}.{key}": torch.full_like(value, index + 1.0)
                for key, value in layer.state_dict().items()
            }
        )
    save_file(state, path)
    return state


def test_apply_distilbert_husk_loads_embeddings_and_all_layers(tmp_path: Path) -> None:
    model = TinyDistilBert()
    path = tmp_path / "husk.safetensors"
    expected = make_husk(model, path)

    apply_distilbert_husk(model, path)

    assert torch.equal(model.embeddings.word_embeddings.weight, expected["word_embeddings"])
    for index, layer in enumerate(model.transformer.layer):
        for key, value in layer.state_dict().items():
            assert torch.equal(value, expected[f"layer.{index}.{key}"])


def test_apply_distilbert_husk_can_preserve_word_embeddings(tmp_path: Path) -> None:
    model = TinyDistilBert()
    original = model.embeddings.word_embeddings.weight.detach().clone()
    path = tmp_path / "husk.safetensors"
    make_husk(model, path)

    apply_distilbert_husk(model, path, load_word_embeddings=False)

    assert torch.equal(model.embeddings.word_embeddings.weight, original)


def test_apply_distilbert_husk_rejects_incomplete_state(tmp_path: Path) -> None:
    model = TinyDistilBert()
    path = tmp_path / "bad.safetensors"
    save_file({"word_embeddings": torch.zeros(7, 3)}, path)

    with pytest.raises(ValueError, match="Incompatible husk"):
        apply_distilbert_husk(model, path)
