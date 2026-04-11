from __future__ import annotations

import torch
from jaxtyping import Float
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-feature scale."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Save configuration so the module is self-describing.
        self.d_model = d_model
        self.eps = eps

        # ``g`` is the learned affine scale applied after normalization.
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # Preserve the caller's dtype so the module behaves well in mixed precision.
        original_dtype = x.dtype

        # Accumulate the RMS statistics in float32 for numerical stability.
        x_float: Float[torch.Tensor, "... d_model"] = x.to(torch.float32)

        # RMSNorm divides by the root mean square over the last feature dimension.
        mean_square: Float[torch.Tensor, "... 1"] = torch.mean(x_float * x_float, dim=-1, keepdim=True)

        # Add epsilon before the square root to avoid division by zero.
        rms: Float[torch.Tensor, "... 1"] = torch.sqrt(mean_square + self.eps)

        # Normalize the activations featurewise.
        normalized: Float[torch.Tensor, "... d_model"] = x_float / rms

        # Apply the learned scale parameter.
        output: Float[torch.Tensor, "... d_model"] = normalized * self.g.to(torch.float32)

        # Convert back to the original dtype expected by downstream layers.
        return output.to(original_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"
