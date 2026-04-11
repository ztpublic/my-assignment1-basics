from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


CheckpointTarget = str | os.PathLike[str] | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointTarget,
) -> None:
    """Serialize model state, optimizer state, and iteration counter."""
    # Package the full training state into one object so restoring is atomic.
    checkpoint = {
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "it": iteration,
    }

    # ``torch.save`` handles both filesystem paths and open binary file objects.
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointTarget,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load a checkpoint into ``model`` and ``optimizer`` and return the stored iteration."""
    # Load the serialized object back into Python.
    checkpoint = torch.load(src)

    # Restore model parameters and buffers.
    model.load_state_dict(checkpoint["model_state"])

    # Restore optimizer hyperparameters and running moments.
    optimizer.load_state_dict(checkpoint["opt_state"])

    # Return the saved training iteration so callers can resume loops cleanly.
    return int(checkpoint["it"])
