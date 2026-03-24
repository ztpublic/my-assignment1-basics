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
        self.to_dff = Linear(d_model, d_ff, device, dtype)
        self.to_dff_gate = Linear(d_model, d_ff, device, dtype)
        self.to_dmodel = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        pass
