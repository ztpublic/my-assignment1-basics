import torch

from cs336_basics.attention import RopeMultiHeadAttention
from cs336_basics.ffn import SwiGLU
from cs336_basics.norm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()
        self.attention = RopeMultiHeadAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff)
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, positions):
        y = x + self.attention(self.attention_norm(x), positions)
        y = y + self.ffn(self.ffn_norm(y))

        return y