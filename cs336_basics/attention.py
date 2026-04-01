from jaxtyping import Float, Bool
from torch import Tensor
import math
import torch

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