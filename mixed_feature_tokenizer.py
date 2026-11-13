"""Prototype for future mixed-schema datasets. Not used in Experiment 02.

Unknown/missing categorical values are coded as 0. Observed values for
categorical field j are 1..K_j; distinct fields have disjoint embedding rows.
"""
from collections.abc import Sequence
import torch
from torch import nn


class MixedFeatureTokenizer(nn.Module):
    def __init__(self, num_numeric: int, cat_cardinalities: Sequence[int], d_model: int):
        super().__init__()
        if num_numeric < 0 or d_model < 1 or any(k < 1 for k in cat_cardinalities):
            raise ValueError("Invalid input dimensions or categorical cardinalities")
        if num_numeric + len(cat_cardinalities) < 1:
            raise ValueError("At least one input feature is required")
        self.num_numeric = num_numeric
        self.cat_cardinalities = tuple(int(k) for k in cat_cardinalities)
        self.num_cat = len(self.cat_cardinalities)
        self.numeric_weights = nn.Parameter(torch.empty(num_numeric, d_model))
        self.numeric_bias = nn.Parameter(torch.empty(num_numeric, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        for parameter in (self.numeric_weights, self.numeric_bias, self.cls_token):
            nn.init.normal_(parameter, std=0.02)
        offsets = [0]
        for k in self.cat_cardinalities[:-1]:
            offsets.append(offsets[-1] + k + 1)
        self.register_buffer("category_offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("category_max_ids", torch.tensor(self.cat_cardinalities, dtype=torch.long))
        self.embedding = (nn.Embedding(sum(k+1 for k in self.cat_cardinalities), d_model)
                          if self.num_cat else None)
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        if x_numeric.ndim != 2 or x_numeric.shape[1] != self.num_numeric:
            raise ValueError("Unexpected numeric input shape")
        if x_categorical.ndim != 2 or x_categorical.shape != (x_numeric.shape[0], self.num_cat):
            raise ValueError("Unexpected categorical input shape")
        if x_categorical.dtype != torch.long:
            raise TypeError("Categorical IDs must be torch.long")
        if self.num_cat and ((x_categorical < 0).any().item()
                             or (x_categorical > self.category_max_ids).any().item()):
            raise ValueError("Categorical ID out of range; encode missing/unseen as 0")
        tokens = []
        if self.num_numeric:
            tokens.append(x_numeric.unsqueeze(-1)*self.numeric_weights + self.numeric_bias)
        if self.num_cat:
            tokens.append(self.embedding(x_categorical+self.category_offsets))
        return torch.cat([self.cls_token.expand(x_numeric.shape[0], -1, -1), *tokens], dim=1)
