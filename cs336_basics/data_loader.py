from __future__ import annotations

import numpy.typing as npt
import torch
from jaxtyping import Int


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[
    Int[torch.Tensor, " batch_size context_length"],
    Int[torch.Tensor, " batch_size context_length"],
]:
    """Sample a random next-token-prediction batch from a 1D token dataset."""
    # The implementation assumes a flat token stream.
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be 1D, got shape {dataset.shape}")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # We need one extra token beyond the input window to form the target window.
    if dataset.shape[0] <= context_length:
        raise ValueError("dataset must be longer than context_length")

    # The last valid start index is ``len(dataset) - context_length - 1``.
    num_valid_starts = dataset.shape[0] - context_length

    # Sample one random start index per example directly on the target device.
    start_indices: Int[torch.Tensor, " batch_size"] = torch.randint(
        0,
        num_valid_starts,
        (batch_size,),
        device=device,
    )

    # Move the flat dataset into a tensor once so indexed slicing stays in PyTorch.
    dataset_tensor: Int[torch.Tensor, " dataset_length"] = torch.as_tensor(
        dataset,
        dtype=torch.long,
        device=device,
    )

    # Build the offsets ``[0, 1, ..., context_length - 1]`` used for every row.
    offsets: Int[torch.Tensor, " context_length"] = torch.arange(context_length, device=device)

    # Broadcast addition turns starts + offsets into a matrix of token positions.
    input_positions: Int[torch.Tensor, " batch_size context_length"] = start_indices[:, None] + offsets

    # Inputs are the selected windows themselves.
    inputs: Int[torch.Tensor, " batch_size context_length"] = dataset_tensor[input_positions]

    # Targets are the same windows shifted one token to the right.
    targets: Int[torch.Tensor, " batch_size context_length"] = dataset_tensor[input_positions + 1]

    return inputs, targets
