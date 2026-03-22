from torch import nn
import torch
import math


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(
            torch.fill(torch.empty(d_model, device=device, dtype=dtype), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        squr_mean = torch.einsum("...d,...d->...", x, x) / x.size(-1)

        rms = torch.sqrt(squr_mean + self.eps)

        devision = x / rms.unsqueeze(-1)

        out = torch.einsum("...d, d->...d", devision, self.g)

        out = out.to(in_dtype)

        return out
