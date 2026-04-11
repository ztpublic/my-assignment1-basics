from __future__ import annotations

import torch
from jaxtyping import Float, Int


class RotaryPositionalEmbedding(torch.nn.Module):
    """Apply rotary position embeddings to query or key vectors."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        # RoPE rotates pairs of channels, so the head dimension must be even.
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for rotary embeddings")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # ``pair_indices`` enumerates the complex-valued frequency bands.
        pair_indices = torch.arange(d_k // 2, device=device, dtype=torch.float32)

        # ``positions`` enumerates token positions from 0 to ``max_seq_len - 1``.
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)

        # Each frequency band uses a different inverse timescale.
        angles = positions / (theta ** (2.0 * pair_indices / d_k))

        # Cache cosines so the forward pass only performs indexing and mixing.
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)

        # Cache sines alongside the cosines for the odd channels.
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        # Split even and odd channels so each adjacent pair can be rotated together.
        x_even: Float[torch.Tensor, "... seq_len half_d_k"] = x[..., 0::2]
        x_odd: Float[torch.Tensor, "... seq_len half_d_k"] = x[..., 1::2]

        # Gather the precomputed trigonometric values for the requested positions.
        cos: Float[torch.Tensor, "... seq_len half_d_k"] = self.cos_cache[token_positions].unsqueeze(-3)
        sin: Float[torch.Tensor, "... seq_len half_d_k"] = self.sin_cache[token_positions].unsqueeze(-3)

        # Apply the 2D rotation to each even/odd channel pair.
        rotated_even: Float[torch.Tensor, "... seq_len half_d_k"] = x_even * cos - x_odd * sin
        rotated_odd: Float[torch.Tensor, "... seq_len half_d_k"] = x_even * sin + x_odd * cos

        # Interleave the rotated pairs back into the original last-dimension layout.
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
