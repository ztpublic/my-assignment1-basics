from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import nn


def _truncated_normal_parameter(shape: tuple[int, ...], *, std: float, device=None, dtype=None) -> nn.Parameter:
    """Create a parameter initialized from a truncated normal distribution."""
    # Allocate the raw storage first so initialization happens in-place.
    parameter = torch.empty(shape, device=device, dtype=dtype)

    # Use symmetric truncation at three standard deviations, which matches the
    # initialization style used throughout the assignment.
    nn.init.trunc_normal_(parameter, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    return nn.Parameter(parameter)


class Linear(nn.Module):
    """A bias-free linear layer implemented as ``x @ W^T``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Record the public shape metadata for debugging and repr output.
        self.in_features = in_features
        self.out_features = out_features

        # Xavier-style variance keeps the forward signal roughly stable.
        std = math.sqrt(2.0 / (in_features + out_features))

        # Store weights in the PyTorch-conventional shape ``[out_features, in_features]``.
        self.weight = _truncated_normal_parameter(
            (out_features, in_features),
            std=std,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "... out_features"]:
        # The final dimension of ``x`` is the feature dimension to project.
        return x @ self.weight.T

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"
