from torch import nn
import torch

from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
        
