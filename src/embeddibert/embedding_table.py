from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def render_wordpiece(token: str, policy: str = "surface") -> str:
    """Turn a tokenizer vocabulary item into the single text sent to Qwen."""
    if policy == "raw":
        return token
    if policy != "surface":
        raise ValueError(f"Unsupported rendering policy: {policy}")
    if token.startswith("##") and len(token) > 2:
        return token[2:]
    return token


def last_token_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool the last non-padding token for either left or right padding."""
    if bool(torch.all(attention_mask[:, -1] == 1)):
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.long().sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, sequence_lengths]


@dataclass
class EmbeddingTableResult:
    table: torch.Tensor
    tokens: list[str]
    rendered_tokens: list[str]


@torch.inference_mode()
def build_qwen_embedding_table(
    *,
    bert_tokenizer,
    qwen_tokenizer,
    qwen_model,
    output_dimension: int,
    batch_size: int,
    device: torch.device,
    token_rendering: str,
    max_vocab_tokens: int | None = None,
) -> EmbeddingTableResult:
    vocab_size = len(bert_tokenizer)
    limit = vocab_size if max_vocab_tokens is None else min(max_vocab_tokens, vocab_size)
    tokens = bert_tokenizer.convert_ids_to_tokens(list(range(limit)))
    rendered = [render_wordpiece(token, token_rendering) for token in tokens]
    rows: list[torch.Tensor] = []

    qwen_model.eval()
    for start in range(0, limit, batch_size):
        texts = rendered[start : start + batch_size]
        encoded = qwen_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = qwen_model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        if pooled.shape[-1] < output_dimension:
            raise ValueError(
                f"Qwen hidden size {pooled.shape[-1]} is smaller than requested "
                f"dimension {output_dimension}"
            )
        # MRL truncation happens before normalization.
        pooled = F.normalize(pooled[:, :output_dimension].float(), p=2, dim=-1)
        rows.append(pooled.cpu())

    return EmbeddingTableResult(torch.cat(rows, dim=0), tokens, rendered)


def save_embedding_table(result: EmbeddingTableResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"word_embeddings": result.table.contiguous().to(torch.float16)},
        output_dir / "qwen_word_embeddings.safetensors",
    )
    manifest = [
        {"id": index, "token": token, "rendered": rendered}
        for index, (token, rendered) in enumerate(zip(result.tokens, result.rendered_tokens))
    ]
    (output_dir / "token_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def table_statistics(table: torch.Tensor) -> dict[str, float | int]:
    norms = table.float().norm(dim=-1)
    return {
        "rows": table.shape[0],
        "dimensions": table.shape[1],
        "mean": table.float().mean().item(),
        "std": table.float().std().item(),
        "mean_row_norm": norms.mean().item(),
        "min_row_norm": norms.min().item(),
        "max_row_norm": norms.max().item(),
    }
