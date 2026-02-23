from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the longest tokens in a vocab.json file.")
    parser.add_argument("vocab_path", type=Path, help="Path to vocab.json.")
    parser.add_argument("--top-k", type=int, default=10, help="How many tokens to print.")
    return parser.parse_args()


def load_token_id_pairs(vocab_path: Path) -> list[tuple[str, int]]:
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected vocab.json to be a JSON object.")

    # Common GPT-2-style format: token -> id
    if all(isinstance(k, str) and isinstance(v, int) for k, v in data.items()):
        return list(data.items())

    # Fallback: id -> token
    if all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        pairs: list[tuple[str, int]] = []
        for k, v in data.items():
            try:
                token_id = int(k)
            except ValueError as exc:
                raise ValueError("Detected id->token format, but found non-integer id keys.") from exc
            pairs.append((v, token_id))
        return pairs

    raise ValueError("Unsupported vocab.json format.")


def main() -> None:
    args = parse_args()
    token_id_pairs = load_token_id_pairs(args.vocab_path)

    longest = sorted(
        token_id_pairs,
        key=lambda item: (len(item[0]), item[0], item[1]),
        reverse=True,
    )[: args.top_k]

    for rank, (token, token_id) in enumerate(longest, start=1):
        print(f"{rank:>2}. len={len(token):>4} id={token_id:>6} token={token!r}")


if __name__ == "__main__":
    main()
