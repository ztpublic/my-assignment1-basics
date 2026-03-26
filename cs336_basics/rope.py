import torch


class RotaryPositionalEmbedding(torch.nn.Modules):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__(self)
        self.theta = theta

    def angle(self, i, k, d) -> float: 
        return i / (self.theta ** ((2 * k - 2)/ d))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

