"""The two feed-forward alternatives. Neither has dropout."""

import torch
from torch import nn
from torch.nn import functional as F


class GELUFFN(nn.Module):
    """Standard Transformer FFN: W_down(GELU(W_up(x)))."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.activation(self.up(x)))


class SwiGLUFFN(nn.Module):
    """SwiGLU: W_down(SiLU(W_gate(x)) * W_up(x))."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim)
        self.up = nn.Linear(d_model, hidden_dim)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
