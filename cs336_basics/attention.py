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


#   1. cs336_basics/attention.py:30 iterate over self.num_heads, which is an int.
#      for i in self.num_heads will fail immediately. This needs
#      range(self.num_heads).
#   2. cs336_basics/attention.py:30 use torch.nn.ParameterList to store
#      Linear(...) modules. ParameterList is for nn.Parameters, not submodules.
#      This should be torch.nn.ModuleList(...) if you keep the per-head design.
#   3. cs336_basics/attention.py:32 defines self.v_k_list, but cs336_basics/
#      attention.py:51 reads self.w_v_list. That attribute does not exist.
#   4. cs336_basics/attention.py:53 passes w_q, w_k, and w_v directly into
#      scaled_dot_product_attention. Those are projection layers, not projected
#      tensors. You need something like Q = w_q(x), K = w_k(x), V = w_v(x) first.
#   5. cs336_basics/attention.py:37 chunks x by self.d_k, not by self.num_heads.
#      That gives the wrong number of chunks. More importantly, standard MHA does
#      not split the raw input first in this way; it projects first, then
#      reshapes/splits into heads.
#   6. cs336_basics/attention.py:48 ignore head entirely. Even if the chunking
#      were right, the loop never uses the chunked data.
#   7. cs336_basics/attention.py:41 create the mask on the default CPU device. If
#      x is on GPU, this will device-mismatch. The mask should be created on
#      x.device.
#   8. cs336_basics/attention.py:29 should validate d_model % num_heads == 0.
#      Without that, invalid configurations silently produce broken head sizes.

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.w_q_list = torch.nn.ModuleList([Linear(self.d_model, self.d_k, device) for i in range(self.num_heads)])
        self.w_k_list = torch.nn.ModuleList([Linear(self.d_model, self.d_k, device) for i in range(self.num_heads)])
        self.v_k_list = torch.nn.ModuleList([Linear(self.d_model, self.d_k, device) for i in range(self.num_heads)])
        self.w_o = Linear(self.d_model, self.d_model, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        heads = torch.chunk(x, self.d_k, dim=-1)

        seq_len = x.size(-2)

        i = torch.arange(seq_len).unsqueeze(1)
        j = torch.arange(seq_len).unsqueeze(0)

        mask = torch.where(j > i, False, True)

        v_out_list = []

        for idx, head in enumerate(heads):
            w_q = self.w_q_list[idx]
            w_k = self.w_k_list[idx]
            w_v = self.w_v_list[idx]

            v_out = scaled_dot_product_attention(w_q, w_k, w_v, mask)
            v_out_list.append(v_out)

        cat = torch.cat(v_out_list, dim=-1)

        out = self.w_o(cat)
        return out