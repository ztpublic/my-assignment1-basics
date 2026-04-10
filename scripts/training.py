from cs336_basics.data_loader import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.softmax import softmax
from cs336_basics.transformer import TransformerLM, TransformerLMConfig
import torch
import numpy as np


def main():
    config = TransformerLMConfig(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
    )

    arr = np.load("data/tiny-stories-10000-tokenized.npy")

    model = TransformerLM.from_config(config, device="mps", dtype=torch.float32)
    optimizer = AdamW(
        model.parameters(), 
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    for i in range(100):
        optimizer.zero_grad()
        inputs, targets = get_batch(arr, 32, 256, "mps")
        out = model(inputs)
        loss = cross_entropy(out, targets)
        loss.backward()
        optimizer.step()

    model_state = model.state_dict()
    torch.save(model_state, "./data/model")


if __name__ == "__main__":
    main()
