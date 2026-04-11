from __future__ import annotations

import torch
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"],
    targets: Int[torch.Tensor, " batch_size"],
) -> Float[torch.Tensor, ""]:
    """Compute the mean negative log-likelihood for a batch of logits.

    This mirrors the core math inside PyTorch's cross-entropy implementation while
    keeping the steps explicit for learning purposes.
    """
    # We expect one target class index per row of logits.
    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape [batch_size, vocab_size], got {tuple(inputs.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape [batch_size], got {tuple(targets.shape)}")
    if inputs.size(0) != targets.size(0):
        raise ValueError("inputs and targets must agree on batch size")

    # Build the row indices used to extract the gold-class logit from each example.
    batch_indices: Int[torch.Tensor, " batch_size"] = torch.arange(inputs.size(0), device=inputs.device)

    # Gather the logit assigned to the correct target class.
    target_logits: Float[torch.Tensor, " batch_size"] = inputs[batch_indices, targets]

    # ``logsumexp`` computes log(sum(exp(logits))) in a stable way.
    normalizer: Float[torch.Tensor, " batch_size"] = torch.logsumexp(inputs, dim=-1)

    # Cross-entropy for each example is ``-log p(target)``.
    per_example_loss: Float[torch.Tensor, " batch_size"] = normalizer - target_logits

    # Return the scalar batch mean.
    return per_example_loss.mean()
