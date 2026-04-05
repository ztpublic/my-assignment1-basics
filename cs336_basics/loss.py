from jaxtyping import Int, Float
import torch

# old bad implementation:
    # inputs_soft = softmax(inputs, -1)
    # inputs_selected = inputs_soft[torch.arange(inputs_soft.size(0)), targets]
    # loss = - torch.log(inputs_selected)
    # return loss.mean()

def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"],
    targets: Int[torch.Tensor, " batch_size"],
):
    batch_indices = torch.arange(inputs.size(0), device=inputs.device)
    target_logits = inputs[batch_indices, targets]
    loss = -target_logits + torch.logsumexp(inputs, dim=-1)
    return loss.mean()
