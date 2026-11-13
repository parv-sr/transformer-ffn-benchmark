"""Experiment 03 model: original encoder blocks + feature-specific mixed tokenizer."""
from __future__ import annotations

from models.common import FeatureTransformerClassifier
from models.ffn import GELUFFN, SwiGLUFFN
from mixed_feature_tokenizer import MixedFeatureTokenizer


class MixedBinaryTransformer(FeatureTransformerClassifier):
    def __init__(self, *, num_numeric: int, cat_cardinalities: tuple[int, ...],
                 d_model: int, depth: int, num_heads: int, kind: str):
        if kind not in ("silu", "swiglu"):
            raise ValueError("Expected silu or swiglu")
        hidden_dim = 4 * d_model if kind == "silu" else round(8 * d_model / 3)
        factory = (lambda d: GELUFFN(d, hidden_dim, activation="silu")
                   if kind == "silu" else SwiGLUFFN(d, hidden_dim))
        super().__init__(num_features=num_numeric + len(cat_cardinalities),
                         d_model=d_model, depth=depth, num_heads=num_heads,
                         ffn_factory=factory)
        self.tokenizer = MixedFeatureTokenizer(num_numeric, cat_cardinalities, d_model)

    def forward(self, x_numeric, x_categorical):
        x = self.tokenizer(x_numeric, x_categorical)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x[:, 0]))
