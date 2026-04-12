import argparse
import sys
from dataclasses import asdict

import numpy as np
import torch
import wandb

from cs336_basics.data_loader import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import TransformerLM, TransformerLMConfig


def resolve_device(requested_device: str) -> str:
    """Resolve a requested training device into a concrete torch device string."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install/environment")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch install/environment")

    return str(torch.device(requested_device))


def accelerator_diagnostics() -> dict[str, object]:
    """Capture the runtime accelerator state that affects device selection."""
    return {
        "torch_version": torch.__version__,
        "torch_cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_available": torch.backends.mps.is_available(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CS336 basics language model.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device to use. Defaults to auto (CUDA, then MPS, then CPU).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TransformerLMConfig(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
    )

    batch_size = 32
    num_steps = 5000
    learning_rate = 1e-3
    weight_decay = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    device = resolve_device(args.device)
    diagnostics = accelerator_diagnostics()

    print(
        "accelerator diagnostics:",
        {**diagnostics, "requested_device": args.device, "selected_device": device},
        file=sys.stderr,
    )
    if args.device == "auto" and device == "cpu" and not diagnostics["cuda_available"]:
        print(
            "warning: training fell back to CPU because CUDA is unavailable in this Python environment. "
            "If you expect GPU training, check that this virtualenv has a CUDA-enabled PyTorch build.",
            file=sys.stderr,
        )

    arr = np.load("data/tiny-stories-10000-tokenized.npy")

    run = wandb.init(
        project="cs336-basics",
        config={
            **asdict(config),
            "batch_size": batch_size,
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "device": device,
            **diagnostics,
        },
    )

    model = TransformerLM.from_config(config, device=device, dtype=torch.float32)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    try:
        for step in range(num_steps):
            optimizer.zero_grad()
            inputs, targets = get_batch(arr, batch_size, config.context_length, device)
            logits = model(inputs)
            loss = cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item()}, step=step)

        model_path = "./data/model.pt"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
