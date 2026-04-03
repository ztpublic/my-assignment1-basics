from jaxtyping import Float, Bool
from torch import Tensor
import math
import torch

from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    K_T = K.transpose(-1, -2)
    d_k = Q.size(-1)
    before_softmax = (Q @ K_T) / math.sqrt(d_k)
    if mask is not None:
        mask2 = torch.where(mask, 0, float('-inf'))
        before_softmax = before_softmax + mask2
    soft = softmax(before_softmax, -1)
    out = soft @ V
    return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.w_q_list = torch.nn.ModuleList([Linear(self.d_model, self.d_k, device) for i in range(self.num_heads)])
        self.w_k_list = torch.nn.ModuleList([Linear(self.d_model, self.d_k, device) for i in range(self.num_heads)])
        self.w_v_list = torch.nn.ModuleList([Linear(self.d_model, self.d_k, device) for i in range(self.num_heads)])
        self.w_o = Linear(self.d_model, self.d_model, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(-2)

        i = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j = torch.arange(seq_len, device=x.device).unsqueeze(0)

        mask = torch.where(j > i, False, True)

        v_out_list = []

        for idx in range(self.num_heads):
            w_q = self.w_q_list[idx]
            w_k = self.w_k_list[idx]
            w_v = self.w_v_list[idx]

            v_out = scaled_dot_product_attention(w_q(x), w_k(x), w_v(x), mask)
            v_out_list.append(v_out)

        cat = torch.cat(v_out_list, dim=-1)

        out = self.w_o(cat)
        return out