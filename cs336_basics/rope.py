import torch
from jaxtyping import Float, Int

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        k = torch.arange(d_k // 2, device=device).unsqueeze(0)
        i = torch.arange(max_seq_len, device=device).unsqueeze(1)

        phi = i / (self.theta ** (2 * k / d_k))

        cos_cache: Float[torch.Tensor, "max_seq_len half_d_k"] = torch.cos(phi)
        sin_cache: Float[torch.Tensor, "max_seq_len half_d_k"] = torch.sin(phi)
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)


    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions:  Int[torch.Tensor, "... seq_len"]) -> torch.Tensor:
        x_even: Float[torch.Tensor, "... seq_len half_d_k"] = x[..., 0::2]
        x_odd: Float[torch.Tensor, "... seq_len half_d_k"] = x[..., 1::2]

        cos_cache: Float[torch.Tensor, "seq_len half_d_k"] = self.get_buffer("cos_cache")
        sin_cache: Float[torch.Tensor, "seq_len half_d_k"] = self.get_buffer("sin_cache")

        # Preserve broadcasting across any extra dimensions before seq_len,
        # including the attention head axis used by batched multi-head attention.
        cos: Float[torch.Tensor, "..., seq_len half_d_k"] = cos_cache[token_positions].unsqueeze(-3)
        sin: Float[torch.Tensor, "..., seq_len half_d_k"] = sin_cache[token_positions].unsqueeze(-3)

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_even_rot, x_odd_rot), dim=-1).flatten(-2)

        return x_rot
