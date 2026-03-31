import torch

def softmax(x: torch.Tensor, dim: int):
    max = x.max(dim=dim, keepdim=True).values
    sub = x - max
    exp = torch.exp(sub)
    exp_sum = exp.sum(dim=dim, keepdim=True)

    return exp / exp_sum