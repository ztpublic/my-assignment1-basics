import argparse
from pathlib import Path
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer


DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
DEFAULT_VOCAB_PATH = Path("data/vocab.json")
DEFAULT_MERGES_PATH = Path("data/merges.txt")
DEFAULT_OUTPUT_PATH = Path("data/tiny-stories-10000-tokenized.npy")
DEFAULT_LOG_INTERVAL_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--log-interval", type=float, default=DEFAULT_LOG_INTERVAL_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    tokenizer_load_start = time.perf_counter()
    tokenizer = Tokenizer.from_files(
        str(args.vocab),
        str(args.merges),
        ["<|endoftext|>"],
    )
    print(f"[load] tokenizer ready in {time.perf_counter() - tokenizer_load_start:.2f}s")

    read_start = time.perf_counter()
    with args.input.open("r", encoding="utf-8") as f:
        content = f.read()
    total_bytes = args.input.stat().st_size
    print(f"[load] read {total_bytes / (1024 * 1024):.2f} MiB in {time.perf_counter() - read_start:.2f}s")

    encode_start = time.perf_counter()
    last_log_time = encode_start
    bytes_processed = 0
    token_count = 0
    pretoken_count = 0
    encoded: list[int] = []

    for pre in tokenizer._pre_token_iter([content]):
        pretoken_count += 1
        bytes_processed += len(pre)

        if pre in tokenizer.special_tokens:
            encoded.append(tokenizer.special_token_dict[pre])
            token_count += 1
        else:
            piece_ids = tokenizer._encode_piece(pre)
            encoded.extend(piece_ids)
            token_count += len(piece_ids)

        now = time.perf_counter()
        if now - last_log_time >= args.log_interval:
            elapsed = now - encode_start
            mib_per_second = (bytes_processed / (1024 * 1024)) / elapsed if elapsed else 0.0
            tokens_per_second = token_count / elapsed if elapsed else 0.0
            progress = (bytes_processed / total_bytes) * 100 if total_bytes else 100.0
            print(
                "[encode] "
                f"{progress:6.2f}% "
                f"pretokens={pretoken_count:,} "
                f"tokens={token_count:,} "
                f"speed={tokens_per_second:,.0f} tok/s "
                f"{mib_per_second:,.2f} MiB/s"
            )
            last_log_time = now

    encode_elapsed = time.perf_counter() - encode_start
    print(
        "[encode] done "
        f"pretokens={pretoken_count:,} "
        f"tokens={token_count:,} "
        f"elapsed={encode_elapsed:.2f}s "
        f"avg_speed={token_count / encode_elapsed:,.0f} tok/s"
    )
    print(f"[encode] cache={tokenizer._encode_piece.cache_info()}")

    save_start = time.perf_counter()
    arr = np.array(encoded, dtype=np.int32)
    np.save(args.output, arr)
    print(f"[save] wrote {args.output} in {time.perf_counter() - save_start:.2f}s")
    print(f"[total] finished in {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()
