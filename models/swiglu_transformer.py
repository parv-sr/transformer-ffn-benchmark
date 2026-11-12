"""Model B: SAME encoder, but its FFN is SwiGLU instead of GELU MLP."""

from .common import FeatureTransformerClassifier
from .ffn import SwiGLUFFN


class SwiGLUTransformer(FeatureTransformerClassifier):
    def __init__(
        self,
        *,
        num_features: int = 30,
        d_model: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        ffn_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            num_features=num_features,
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            ffn_factory=lambda size: SwiGLUFFN(size, ffn_hidden_dim),
        )
