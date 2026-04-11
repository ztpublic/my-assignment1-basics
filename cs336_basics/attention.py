from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.softmax import softmax


def _causal_mask(seq_len: int, device: torch.device) -> Bool[Tensor, " seq_len seq_len"]:
    """Return a lower-triangular boolean causal mask."""
    # ``torch.tril`` keeps the diagonal and all positions to its left.
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Compute scaled dot-product attention."""
    # Transpose keys so each query can dot-product against every key vector.
    key_transpose: Float[Tensor, " ... d_k keys"] = K.transpose(-1, -2)

    # The scale factor uses the key/query width.
    d_k = Q.size(-1)

    # Raw attention scores are scaled dot products.
    attention_scores: Float[Tensor, " ... queries keys"] = (Q @ key_transpose) / math.sqrt(d_k)

    # Invalid attention locations receive ``-inf`` so softmax maps them to zero.
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

    # Normalize scores into attention probabilities.
    attention_weights: Float[Tensor, " ... queries keys"] = softmax(attention_scores, -1)

    # Use the probabilities to mix the value vectors.
    return attention_weights @ V


class MultiHeadAttention(torch.nn.Module):
    """Causal self-attention with packed Q/K/V projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        # Each head must receive an integer-sized slice of the model dimension.
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Project into the packed query, key, and value spaces.
        self.w_q = Linear(d_model, d_model, device)
        self.w_k = Linear(d_model, d_model, device)
        self.w_v = Linear(d_model, d_model, device)

        # Project the concatenated head outputs back to model width.
        self.w_o = Linear(d_model, d_model, device)

    def _split_heads(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
    ) -> Float[Tensor, "... num_heads sequence_length d_k"]:
        # Reshape ``[..., seq_len, d_model]`` into ``[..., num_heads, seq_len, d_k]``.
        return x.reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)

    def _merge_heads(
        self,
        x: Float[Tensor, "... num_heads sequence_length d_k"],
    ) -> Float[Tensor, "... sequence_length d_model"]:
        # Invert ``_split_heads`` by moving the head axis back next to the feature axis.
        return x.transpose(-2, -3).reshape(*x.shape[:-3], x.shape[-2], self.d_model)

    def forward(self, x: Float[Tensor, "... sequence_length d_model"]) -> Float[Tensor, "... sequence_length d_model"]:
        # The input is an autoregressive sequence, so we build a causal mask.
        seq_len = x.size(-2)
        mask: Bool[Tensor, " sequence_length sequence_length"] = _causal_mask(seq_len, x.device)

        # Project the model states into Q, K, and V tensors.
        q: Float[Tensor, "... sequence_length d_model"] = self.w_q(x)
        k: Float[Tensor, "... sequence_length d_model"] = self.w_k(x)
        v: Float[Tensor, "... sequence_length d_model"] = self.w_v(x)

        # Split the packed projections into separate attention heads.
        q_heads: Float[Tensor, "... num_heads sequence_length d_k"] = self._split_heads(q)
        k_heads: Float[Tensor, "... num_heads sequence_length d_k"] = self._split_heads(k)
        v_heads: Float[Tensor, "... num_heads sequence_length d_k"] = self._split_heads(v)

        # Run attention independently per head.
        attended_values: Float[Tensor, "... num_heads sequence_length d_k"] = scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            mask,
        )

        # Concatenate the head outputs back together.
        merged: Float[Tensor, "... sequence_length d_model"] = self._merge_heads(attended_values)

        # Apply the final output projection.
        return self.w_o(merged)


class RopeMultiHeadAttention(torch.nn.Module):
    """Causal self-attention with rotary positional embeddings."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        # RoPE uses the same per-head dimensionality constraint as vanilla MHA.
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # The projections are identical to standard MHA.
        self.w_q = Linear(d_model, d_model, device)
        self.w_k = Linear(d_model, d_model, device)
        self.w_v = Linear(d_model, d_model, device)
        self.w_o = Linear(d_model, d_model, device)

        # RoPE is applied per attention head.
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)

    def _split_heads(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
    ) -> Float[Tensor, "... num_heads sequence_length d_k"]:
        # Reshape ``[..., seq_len, d_model]`` into ``[..., num_heads, seq_len, d_k]``.
        return x.reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)

    def _merge_heads(
        self,
        x: Float[Tensor, "... num_heads sequence_length d_k"],
    ) -> Float[Tensor, "... sequence_length d_model"]:
        # Invert ``_split_heads`` by restoring the packed model dimension.
        return x.transpose(-2, -3).reshape(*x.shape[:-3], x.shape[-2], self.d_model)

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        # Build the standard causal mask for autoregressive decoding.
        seq_len = x.size(-2)
        mask: Bool[Tensor, " sequence_length sequence_length"] = _causal_mask(seq_len, x.device)

        # If the caller does not provide positions, assume a standard contiguous range.
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).expand(*x.shape[:-2], seq_len)

        # Project input states into packed Q/K/V tensors.
        q: Float[Tensor, "... sequence_length d_model"] = self.w_q(x)
        k: Float[Tensor, "... sequence_length d_model"] = self.w_k(x)
        v: Float[Tensor, "... sequence_length d_model"] = self.w_v(x)

        # Split projections into heads before applying RoPE.
        q_heads: Float[Tensor, "... num_heads sequence_length d_k"] = self._split_heads(q)
        k_heads: Float[Tensor, "... num_heads sequence_length d_k"] = self._split_heads(k)
        v_heads: Float[Tensor, "... num_heads sequence_length d_k"] = self._split_heads(v)

        # RoPE rotates queries and keys but leaves values unchanged.
        q_rotated: Float[Tensor, "... num_heads sequence_length d_k"] = self.rope(q_heads, token_positions)
        k_rotated: Float[Tensor, "... num_heads sequence_length d_k"] = self.rope(k_heads, token_positions)

        # Compute causal attention using the rotated queries and keys.
        attended_values: Float[Tensor, "... num_heads sequence_length d_k"] = scaled_dot_product_attention(
            q_rotated,
            k_rotated,
            v_heads,
            mask,
        )

        # Merge the attended head outputs back into the model dimension.
        merged: Float[Tensor, "... sequence_length d_model"] = self._merge_heads(attended_values)

        # Finish with the standard output projection.
        return self.w_o(merged)
