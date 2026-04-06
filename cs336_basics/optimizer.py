from collections.abc import Iterable, Callable
from typing import Any, Dict, Tuple
from typing import Optional
import torch
import math

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    norm = torch.zeros(1)
    for p in parameters:
        g = p.grad
        if g is None:
            continue
        norm += g.pow(2).sum()

    norm = torch.sqrt(norm)

    if norm <= max_l2_norm:
        return

    scale = max_l2_norm / norm

    for p in parameters:
        g = p.grad
        if g is None:
            continue
        g.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: (
            Iterable[torch.Tensor]
            | Iterable[Dict[str, Any]]
            | Iterable[Tuple[str, torch.Tensor]]
        ),
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
        eps: float,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps
        }
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros(p.grad.data.shape, device=p.grad.data.device))
                v = state.get("v", torch.zeros(p.grad.data.shape, device=p.grad.data.device))
                t = state.get("t", 1) 

                g = p.grad.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g ** 2

                lr_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

                p.data -= lr_t * (m / (torch.sqrt(v) + eps))

                p.data -= p.data * lr * weight_decay

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss