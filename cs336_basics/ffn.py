from __future__ import annotations

import torch
from jaxtyping import Float
from torch import nn

from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    """A SwiGLU feed-forward network built from three bias-free linear layers."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # ``w1`` produces the gated branch before the SiLU nonlinearity.
        self.w1 = Linear(d_model, d_ff, device, dtype)

        # ``w2`` projects the gated hidden representation back to model width.
        self.w2 = Linear(d_ff, d_model, device, dtype)

        # ``w3`` produces the value branch that is modulated by the gate.
        self.w3 = Linear(d_model, d_ff, device, dtype)

    @staticmethod
    def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        # SiLU is ``x * sigmoid(x)``.
        return x * torch.sigmoid(x)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # Compute the gate branch.
        gate: Float[torch.Tensor, "... d_ff"] = self.silu(self.w1(x))

        # Compute the value branch in parallel.
        values: Float[torch.Tensor, "... d_ff"] = self.w3(x)

        # Multiply gate and values elementwise, then project back down.
        return self.w2(gate * values)
