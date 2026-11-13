"""Model A: pre-LN bidirectional attention + GELU MLP FFN."""

from .common import FeatureTransformerClassifier
from .ffn import GELUFFN


class GELUTransformer(FeatureTransformerClassifier):
    def __init__(
        self,
        *,
        num_features: int = 30,
        d_model: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        ffn_hidden_dim: int = 384,
        mlp_activation: str = "gelu",
    ) -> None:
        super().__init__(
            num_features=num_features,
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            ffn_factory=lambda size: GELUFFN(size, ffn_hidden_dim, activation=mlp_activation),
        )
