from __future__ import annotations

from typing import List

import torch
from jaxtyping import Bool, Float, Int

from cs336_basics.softmax import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def temperature_softmax(
    tensor: Float[torch.Tensor, "... vocab_size"],
    temp: float,
    dim: int,
) -> Float[torch.Tensor, "... vocab_size"]:
    """Apply temperature scaling before softmax."""
    # A non-positive temperature would invert or explode the distribution.
    if temp <= 0:
        raise ValueError("temp must be positive")

    # Scale logits by temperature, then normalize them with our handwritten softmax.
    return softmax(tensor / temp, dim=dim)


def sample_top_p(
    probs: Float[torch.Tensor, "... vocab_size"],
    p: float = 0.9,
) -> Int[torch.Tensor, "..."]:
    """Sample token ids using nucleus sampling from a normalized distribution."""
    # Nucleus sampling requires a valid cumulative probability threshold.
    if not (0 < p <= 1):
        raise ValueError(f"p must be in (0, 1], got {p}")

    # Sort candidates from most likely to least likely.
    sorted_probs: Float[torch.Tensor, "... vocab_size"]
    sorted_indices: Int[torch.Tensor, "... vocab_size"]
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Compute the running cumulative probability mass.
    cumulative_probs: Float[torch.Tensor, "... vocab_size"] = torch.cumsum(sorted_probs, dim=-1)

    # Mark everything after the nucleus cutoff for removal.
    sorted_mask: Bool[torch.Tensor, "... vocab_size"] = cumulative_probs > p

    # Keep the first token that pushes us over the threshold.
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Zero out probabilities outside the nucleus.
    filtered_sorted_probs: Float[torch.Tensor, "... vocab_size"] = sorted_probs.masked_fill(sorted_mask, 0.0)

    # Renormalize over the surviving nucleus candidates.
    filtered_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(dim=-1, keepdim=True)

    # ``torch.multinomial`` expects a 2D matrix, so flatten leading dimensions temporarily.
    sampled_sorted_idx: Int[torch.Tensor, "..."] = torch.multinomial(
        filtered_sorted_probs.reshape(-1, filtered_sorted_probs.size(-1)),
        num_samples=1,
    ).reshape(probs.shape[:-1])

    # Map the sampled sorted indices back into the original vocabulary ids.
    return torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx.unsqueeze(-1)).squeeze(-1)


class Decoder:
    """Convenience wrapper for autoregressive text generation."""

    def __init__(
        self,
        llm: TransformerLM,
        tokenizer: Tokenizer,
        softmax_temp: float,
        top_p: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Keep the language model and tokenizer together.
        self.llm = llm
        self.tokenizer = tokenizer

        # Record the sampling hyperparameters.
        self.softmax_temp = softmax_temp
        self.top_p = top_p

        # Respect an explicit device override, otherwise infer from the model.
        self.device = device if device is not None else next(llm.parameters()).device
        self.dtype = dtype

        # Cache the end-of-text token and context length for generation.
        self.eot_token = tokenizer.encode("<|endoftext|>")[0]
        self.context_length = llm.context_length

    @torch.no_grad()
    def completion(self, prompts: str, max_token_count: int = 100) -> str:
        """Generate a completion string from ``prompts``."""
        # Encode the input prompt once at the start of generation.
        prompt_tokens = self.tokenizer.encode(prompts)

        # Hold the running generated sequence as token ids.
        token_ids: Int[torch.Tensor, " 1 prompt_plus_generated"] = torch.tensor(
            prompt_tokens,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)

        # Collect decoded text fragments as we sample new tokens.
        output: List[str] = []

        while len(output) < max_token_count:
            # Feed only the last ``context_length`` tokens to the model, which is the
            # standard sliding-window approach for finite-context decoders.
            model_input: Int[torch.Tensor, " 1 active_sequence_length"] = token_ids[:, -self.context_length :]

            # Run the model and take the logits for the newest position.
            logits: Float[torch.Tensor, "batch seq vocab_size"] = self.llm(model_input)
            next_token_logits: Float[torch.Tensor, " vocab_size"] = logits[0, -1]

            # Convert logits into a probability distribution with temperature scaling.
            probabilities: Float[torch.Tensor, " vocab_size"] = temperature_softmax(
                next_token_logits,
                self.softmax_temp,
                -1,
            )

            # Sample one next-token id using nucleus sampling.
            sampled_token: Int[torch.Tensor, ""] = sample_top_p(probabilities, self.top_p)
            token: Int = sampled_token.item()

            # Stop generation if we hit the end-of-text token.
            if token == self.eot_token:
                break

            # Decode just the sampled token and append it to the output text pieces.
            output.append(self.tokenizer.decode([token]))

            # Append the sampled token to the running token sequence.
            next_token_tensor: Int[torch.Tensor, " 1 1"] = torch.tensor(
                [[token]],
                device=self.device,
                dtype=torch.long,
            )
            token_ids = torch.cat([token_ids, next_token_tensor], dim=-1)

        # Join the token-level text pieces into the final completion.
        return "".join(output)
