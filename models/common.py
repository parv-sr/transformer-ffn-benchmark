"""Shared encoder architecture. The FFN factory is the ONLY architecture switch."""

from collections.abc import Callable

import torch
from torch import nn


class ScalarFeatureTokenizer(nn.Module):
    """Feature-specific affine tokenization: token_j = x_j * W_j + b_j.

    Every numeric feature becomes one d_model-dimensional token. The learned
    per-feature bias/weight identify which original column produced each token,
    so positional encodings are unnecessary for this unordered feature set.
    """

    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.empty(num_features, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError(
                f"Expected (batch, {self.weight.shape[0]}) features, got {tuple(x.shape)}"
            )
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls, tokens), dim=1)


class PreLayerNormTransformerBlock(nn.Module):
    """x = x + MHA(LN(x)); x = x + FFN(LN(x)). No dropout or mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = ffn_factory(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(x)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
            attn_mask=None,
            key_padding_mask=None,
            is_causal=False,
        )
        x = x + attention_output
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FeatureTransformerClassifier(nn.Module):
    """Encoder-only Transformer with a learned [CLS] token and binary head.

    forward() returns RAW LOGITS for numerically stable CrossEntropyLoss.
    predict_proba() applies the requested softmax during inference/evaluation.
    """

    def __init__(
        self,
        *,
        num_features: int = 30,
        d_model: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        ffn_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.tokenizer = ScalarFeatureTokenizer(num_features, d_model)
        self.blocks = nn.ModuleList(
            [PreLayerNormTransformerBlock(d_model, num_heads, ffn_factory) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        for block in self.blocks:
            x = block(x)
        cls = self.final_norm(x[:, 0])
        return self.head(cls)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).softmax(dim=-1)
