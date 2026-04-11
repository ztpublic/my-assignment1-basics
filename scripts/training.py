from dataclasses import asdict

import numpy as np
import torch
import wandb

from cs336_basics.data_loader import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import TransformerLM, TransformerLMConfig


def main():
    config = TransformerLMConfig(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
    )

    batch_size = 32
    num_steps = 100
    learning_rate = 1e-3
    weight_decay = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8
    device = "mps" if torch.backends.mps.is_available() else "cpu"

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
