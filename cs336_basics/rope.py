import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__(self)
        self.theta = theta
        self.d_k = d_k
        self.register_buffer("rotation_cache", torch.empty(int(d_k / 2), max_seq_len, max_seq_len, device=device))

    def angle(self, i, k, d) -> float: 
        return i / (self.theta ** ((2 * k - 2)/ d))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        

        
        x = torch.arange(self.d_k / 2)
        y = torch.stack(
            [torch.stack([], dim=-1),
            torch.stack([], dim=-1)], dim=-2
        )
        raise NotImplementedError

