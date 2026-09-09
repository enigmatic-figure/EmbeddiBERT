from __future__ import annotations

import torch
from torch import nn


class QwenPairClassifier(nn.Module):
    """Use two Qwen sentence embeddings as DistilBERT input positions."""

    def __init__(self, sequence_classifier, *, cls_token_id: int, sep_token_id: int):
        super().__init__()
        self.distilbert = sequence_classifier.distilbert
        self.pre_classifier = sequence_classifier.pre_classifier
        self.classifier = sequence_classifier.classifier
        self.dropout = sequence_classifier.dropout
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError("left and right must both have shape [batch, hidden]")
        word_table = self.distilbert.embeddings.word_embeddings.weight
        cls = word_table[self.cls_token_id].expand(left.shape[0], -1)
        sep = word_table[self.sep_token_id].expand(left.shape[0], -1)
        inputs_embeds = torch.stack((cls, left, sep, right, sep), dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        hidden = self.distilbert(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        pooled = torch.relu(self.pre_classifier(hidden))
        return self.classifier(self.dropout(pooled))

