from __future__ import annotations

import torch
from jaxtyping import Float


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    """Compute a numerically stable softmax along ``dim``.

    We keep this handwritten instead of delegating to ``torch.softmax`` because the
    assignment is explicitly about understanding the implementation.
    """
    # ``torch.max`` gives us the largest logit per reduction slice.
    max_values: Float[torch.Tensor, "..."] = x.max(dim=dim, keepdim=True).values

    # Subtracting the max keeps the exponentials in a safe numeric range.
    shifted_logits: Float[torch.Tensor, "..."] = x - max_values

    # Exponentiate the centered logits.
    exp_shifted_logits: Float[torch.Tensor, "..."] = torch.exp(shifted_logits)

    # Sum the exponentials over the requested axis.
    normalization: Float[torch.Tensor, "..."] = exp_shifted_logits.sum(dim=dim, keepdim=True)

    # Divide elementwise to obtain a valid probability distribution.
    return exp_shifted_logits / normalization
