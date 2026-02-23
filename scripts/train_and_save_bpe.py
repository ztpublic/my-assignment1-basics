from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cs336_basics.bpe import my_run_train_bpe


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Map each byte (0..255) to a printable GPT-2 unicode surrogate."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE and save vocab/merges to disk.")
    parser.add_argument("input_path", type=Path, help="Path to training text file.")
    parser.add_argument("output_dir", type=Path, help="Directory where vocab.json and merges.txt are saved.")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Total vocabulary size including special tokens.")
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help="Special token to keep atomic. Can be repeated. Default: <|endoftext|>",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of processes for chunk pre-tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_tokens or ["<|endoftext|>"]

    vocab, merges = my_run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        kwargs={"num_processes": args.num_processes},
    )

    byte_to_unicode = gpt2_bytes_to_unicode()

    def token_bytes_to_str(token_bytes: bytes) -> str:
        return "".join(byte_to_unicode[b] for b in token_bytes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.txt"

    gpt2_vocab = {token_bytes_to_str(token_bytes): token_id for token_id, token_bytes in vocab.items()}
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(gpt2_vocab, f, ensure_ascii=False, indent=2)

    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{token_bytes_to_str(left)} {token_bytes_to_str(right)}\n")

    print(f"Saved vocab: {vocab_path}")
    print(f"Saved merges: {merges_path}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")


if __name__ == "__main__":
    main()
