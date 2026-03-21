from torch import nn
import math
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        self.p = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 0, 1, -3, 3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.p[token_ids]