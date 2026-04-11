from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import nn


class Embedding(nn.Module):
    """A minimal learned embedding table.

    The parameter is intentionally named ``p`` to preserve compatibility with the
    existing tests and adapter code in this repository.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Keep the shape metadata around for debugging and repr output.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Allocate the full lookup table once.
        embedding_table = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)

        # Match the original assignment initialization: truncated normal with unit std.
        nn.init.trunc_normal_(embedding_table, mean=0.0, std=1.0, a=-3.0, b=3.0)

        # Register the embedding matrix as a learnable parameter.
        self.p = nn.Parameter(embedding_table)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        # Advanced indexing selects the row for every token id in the input tensor.
        return self.p[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
