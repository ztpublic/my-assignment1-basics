from torch import nn
import math
import torch

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        sd_sqr = 2 / (in_features + out_features)
        sd = math.sqrt(sd_sqr)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                0,
                sd,
                -3 * sd,
                3 * sd,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
