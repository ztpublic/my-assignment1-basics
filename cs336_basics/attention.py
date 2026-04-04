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
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        self.w_q = Linear(self.d_model, self.d_model, device)
        self.w_k = Linear(self.d_model, self.d_model, device)
        self.w_v = Linear(self.d_model, self.d_model, device)

        self.w_o = Linear(self.d_model, self.d_model, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(-2)

        i = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j = torch.arange(seq_len, device=x.device).unsqueeze(0)

        mask = torch.where(j > i, False, True)

        q: Float[Tensor, " ... seq d_model"] = self.w_q(x)
        k: Float[Tensor, " ... seq d_model"] = self.w_k(x)
        v: Float[Tensor, " ... seq d_model"] = self.w_v(x)

        q_reshape = q.reshape(*q.shape[:-1], self.num_heads, -1).transpose(-2, -3)
        k_reshape = k.reshape(*k.shape[:-1], self.num_heads, -1).transpose(-2, -3)
        v_reshape = v.reshape(*v.shape[:-1], self.num_heads, -1).transpose(-2, -3)

        v_out = scaled_dot_product_attention(q_reshape, k_reshape, v_reshape, mask)

        v_out_reshape = v_out.transpose(-2, -3).reshape(*v.shape)
        out = self.w_o(v_out_reshape)
        return out
