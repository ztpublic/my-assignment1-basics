from jaxtyping import Float, Bool
from torch import Tensor
import math
import torch
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    K_T = K.transpose(-1, -2)
    d_k = Q.size(-1)
    seq_len = Q.size(-2)
    before_softmax = (Q @ K_T) / math.sqrt(d_k)
    raise NotImplementedError