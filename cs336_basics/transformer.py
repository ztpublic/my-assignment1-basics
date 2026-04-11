from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int

from cs336_basics.attention import RopeMultiHeadAttention
from cs336_basics.embedding import Embedding
from cs336_basics.ffn import SwiGLU
from cs336_basics.linear import Linear
from cs336_basics.norm import RMSNorm


@dataclass(frozen=True)
class TransformerLMConfig:
    """Configuration object for ``TransformerLM``."""

    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0

    def validate(self) -> None:
        # Each head must receive an integer number of features.
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class TransformerBlock(torch.nn.Module):
    """A pre-norm Transformer block with RoPE attention and SwiGLU FFN."""

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

        # Self-attention sublayer.
        self.attention = RopeMultiHeadAttention(d_model, num_heads, max_seq_len, theta, device)

        # Feed-forward sublayer.
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

        # Independent pre-norms for attention and FFN branches.
        self.attention_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, " batch sequence_length d_model"],
        positions: Int[torch.Tensor, " batch sequence_length"],
    ) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        # Pre-normalize before attention, then add the residual connection.
        attention_residual: Float[torch.Tensor, " batch sequence_length d_model"] = (
            x + self.attention(self.attention_norm(x), positions)
        )

        # Pre-normalize before the FFN, then add the second residual connection.
        return attention_residual + self.ffn(self.ffn_norm(attention_residual))


class TransformerLM(torch.nn.Module):
    """A causal Transformer language model built from handwritten layers."""

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

        # Persist the sequence limit so forward passes can validate their inputs.
        self.context_length = context_length

        # Token embedding lookup.
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Stack of identical Transformer blocks.
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization before projecting into vocabulary logits.
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection into vocabulary space.
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @classmethod
    def from_config(
        cls,
        config: TransformerLMConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TransformerLM":
        # Validate the configuration once at construction time.
        config.validate()

        # Forward the config fields into the concrete module constructor.
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
        # Validate before using the shape arithmetic.
        config.validate()

        # One token embedding matrix.
        embedding_params = config.vocab_size * config.d_model

        # Q, K, V, and output projections inside attention.
        attention_params = 4 * config.d_model * config.d_model

        # SwiGLU has three bias-free linear layers.
        ffn_params = 3 * config.d_model * config.d_ff

        # Each block has two RMSNorm scale vectors.
        block_norm_params = 2 * config.d_model

        # The final RMSNorm adds one more scale vector.
        final_norm_params = config.d_model

        # The output projection is a second embedding-sized matrix.
        return (
            2 * embedding_params
            + config.num_layers * (attention_params + ffn_params + block_norm_params)
            + final_norm_params
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        # Count either trainable parameters only or all registered parameters.
        if trainable_only:
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        x: Int[torch.Tensor, " batch_size sequence_length"],
    ) -> Float[torch.Tensor, " batch_size sequence_length vocab_size"]:
        # Enforce the model's configured context length.
        batch_size, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds model context length {self.context_length}"
            )

        # Build the absolute token positions used by RoPE.
        positions: Int[torch.Tensor, " batch_size sequence_length"] = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        # Convert token ids into dense embeddings.
        hidden_states: Float[torch.Tensor, " batch_size sequence_length d_model"] = self.embedding(x)

        # Pass the sequence through each Transformer block in order.
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, positions)

        # Normalize the final hidden states.
        hidden_states = self.norm(hidden_states)

        # Project into vocabulary logits.
        return self.linear(hidden_states)
