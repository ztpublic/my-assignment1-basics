from __future__ import annotations

import math


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate for iteration ``it`` under warmup + cosine decay."""
    # Negative iteration counts are always a caller bug.
    if it < 0:
        raise ValueError("iteration must be non-negative")

    # A cosine schedule only makes sense when the decay window is at least as long
    # as the warmup window.
    if cosine_cycle_iters < warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than or equal to warmup_iters")

    # If we are still in warmup, linearly interpolate from 0 to the max learning rate.
    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # If there is no cosine region, the schedule stays flat after warmup.
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate if it >= cosine_cycle_iters else max_learning_rate

    # During the cosine window, smoothly decay from max to min.
    if it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_multiplier * (max_learning_rate - min_learning_rate)

    # After the decay window, clamp to the floor learning rate.
    return min_learning_rate
