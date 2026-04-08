from typing import List

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM, TransformerLMConfig
import torch
from jaxtyping import Float, Int

def temperature_softmax(tensor: torch.Tensor, temp: float, dim: int):
    divided = tensor / temp
    exp = torch.exp(divided)
    exp_sum = exp.sum(dim=dim, keepdim=True)
    return exp / exp_sum

def sample_top_p(
    probs: torch.Tensor,
    p: float = 0.9,
) -> torch.Tensor:
    """
    probs: (..., vocab_size), already softmaxed over the last dim
    p: cumulative probability threshold

    returns:
        sampled token indices with shape probs.shape[:-1]
    """
    if not (0 < p <= 1):
        raise ValueError(f"p must be in (0, 1], got {p}")

    # Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Cumulative sum over sorted probabilities
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens once cumulative probability exceeds p
    # Keep the first token that crosses the threshold
    sorted_mask = cum_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Zero out filtered tokens
    filtered_sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

    # Renormalize
    filtered_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(
        dim=-1, keepdim=True
    )

    # Sample from filtered distribution
    sampled_sorted_idx = torch.multinomial(
        filtered_sorted_probs.reshape(-1, filtered_sorted_probs.size(-1)), num_samples=1
    ).reshape(probs.shape[:-1])

    # Map back to original token indices
    sampled_token = torch.gather(
        sorted_indices, dim=-1, index=sampled_sorted_idx.unsqueeze(-1)
    ).squeeze(-1)

    return sampled_token


class Decoder:
    def __init__(
        self,
        llm: TransformerLM,
        tokenizer: Tokenizer,
        softmax_temp: float,
        top_p: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device
        self.softmax_temp = softmax_temp
        self.top_p = top_p
        self.eot_token = tokenizer.encode("<|endoftext|>")[0]
        self.context_length = llm.context_length

    def completion(self, prompts: str, max_token_count: int = 100) -> str:
        prompt_tokens = self.tokenizer.encode(prompts)
        x = torch.tensor(prompt_tokens, device=self.device, dtype=torch.long)
        x.unsqueeze_(0)
        output: List[str] = []
        while True:
            if x.size(-1) > self.context_length:
                break
            o: Float[torch.Tensor, "batch seq vocab_size"] = self.llm(x)
            last = o[0, -1]
            soft = temperature_softmax(last, self.softmax_temp, -1)
            sample = sample_top_p(soft, self.top_p)
            token: Int = sample.item()
            if token == self.eot_token:
                break
            token_str = self.tokenizer.decode([token])
            output.append(token_str)
            if len(output) == max_token_count:
                break
            t = torch.tensor([[token]], device=x.device, dtype=torch.long)
            x = torch.cat([x, t], dim=-1)

        return "".join(output)
