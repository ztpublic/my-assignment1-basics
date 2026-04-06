import numpy.typing
import torch


def get_batch(
    dataset: numpy.typing.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = dataset.shape[0] - context_length
    start_indices = torch.randint(0, max_start, (batch_size,), device=device)

    dataset_tensor = torch.as_tensor(dataset, dtype=torch.long, device=device)
    offsets = torch.arange(context_length, device=device)

    inputs = dataset_tensor[start_indices[:, None] + offsets]
    targets = dataset_tensor[start_indices[:, None] + offsets + 1]

    return inputs, targets
