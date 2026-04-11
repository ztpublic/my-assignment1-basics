from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients in-place so their global L2 norm is at most ``max_l2_norm``."""
    # Gradient clipping with a negative threshold is undefined.
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative")

    # Materialize the gradients once so we can traverse them multiple times safely.
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]

    # If no parameter currently has a gradient, there is nothing to clip.
    if not gradients:
        return

    # Accumulate the squared L2 norm in float32 for stable reduction.
    total_squared_norm = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        gradient_float = gradient.detach().to(torch.float32)
        total_squared_norm += torch.sum(gradient_float * gradient_float)

    # Finish the norm computation with a scalar square root.
    total_norm = torch.sqrt(total_squared_norm)

    # If the norm is already within the threshold, leave the gradients untouched.
    if total_norm <= max_l2_norm:
        return

    # Compute the shared rescaling factor.
    scale = (max_l2_norm / total_norm).to(dtype=gradients[0].dtype)

    # Apply the same multiplicative correction to every gradient tensor.
    for gradient in gradients:
        gradient.mul_(scale)


class AdamW(torch.optim.Optimizer):
    """A small handwritten AdamW optimizer.

    The implementation follows the standard decoupled-weight-decay update:

    1. Update first and second moment estimates.
    2. Apply bias correction.
    3. Take the Adam step.
    4. Apply decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        # Validate scalar hyperparameters early so errors are local and readable.
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")

        # Store defaults in the same shape that PyTorch optimizers expect.
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
        }

        # Delegate parameter-group bookkeeping to ``torch.optim.Optimizer``.
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform one optimization step."""
        loss = None

        # Some optimizers support reevaluating the model under gradients via a closure.
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over parameter groups so each group can have distinct hyperparameters.
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

            # Step each parameter independently using its own optimizer state.
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                # AdamW is conventionally defined for dense gradients.
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                # Create state tensors lazily so the optimizer can be constructed cheaply.
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Update the biased first moment estimate.
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)

                # Update the biased second raw moment estimate.
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                # Compute the standard Adam bias-correction factors.
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                # Convert the second moment into a stable RMS denominator.
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                # Bias-correct the first moment through the step size.
                step_size = lr / bias_correction1

                # Apply the Adam parameter update.
                parameter.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply decoupled weight decay after the adaptive step.
                if weight_decay != 0.0:
                    parameter.add_(parameter, alpha=-lr * weight_decay)

        return loss
