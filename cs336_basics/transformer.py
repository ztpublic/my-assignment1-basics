from dataclasses import dataclass

import torch
from jaxtyping import Int

from cs336_basics.attention import RopeMultiHeadAttention
from cs336_basics.embedding import Embedding
from cs336_basics.ffn import SwiGLU
from cs336_basics.linear import Linear
from cs336_basics.norm import RMSNorm


@dataclass(frozen=True)
class TransformerLMConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention = RopeMultiHeadAttention(d_model, num_heads, max_seq_len, theta, device)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.attention_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)

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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @classmethod
    def from_config(
        cls,
        config: TransformerLMConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TransformerLM":
        config.validate()
        return cls(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def parameter_count_from_config(config: TransformerLMConfig) -> int:
        config.validate()
        embedding_params = config.vocab_size * config.d_model
        attention_params = 4 * config.d_model * config.d_model
        ffn_params = 3 * config.d_model * config.d_ff
        block_norm_params = 2 * config.d_model
        final_norm_params = config.d_model

        return (
            2 * embedding_params
            + config.num_layers * (attention_params + ffn_params + block_norm_params)
            + final_norm_params
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: Int[torch.Tensor, " batch_size sequence_length"]):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        e = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            e = transformer_block(e, positions)
        e = self.norm(e)
        e = self.linear(e)
        return e
