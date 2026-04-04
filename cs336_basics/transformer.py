import torch

from cs336_basics.attention import RopeMultiHeadAttention
from cs336_basics.embedding import Embedding
from cs336_basics.ffn import SwiGLU
from cs336_basics.linear import Linear
from cs336_basics.norm import RMSNorm

from jaxtyping import Int

from cs336_basics.softmax import softmax


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


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: Int[torch.Tensor, " batch_size sequence_length"]):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        e = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            e = transformer_block(e, positions)
        e = self.norm(e)
        e = self.linear(e)
        return e
