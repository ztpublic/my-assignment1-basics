import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re


PRE_TPKEN_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def my_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    cur_token_id = 256

    for special in special_tokens:
        vocab[cur_token_id] = special.encode("utf-8")
        cur_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

    pair_count_map: dict[tuple[bytes, bytes], int] = {}

    pre_token_map: dict[tuple[bytes, ...], int] = {}

    # initial pair count for pre-tokens
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for pre in re.finditer(PRE_TPKEN_PAT, chunk):
                pre_bytes = pre.group(0).encode("utf-8")
                pre_tuple: tuple[bytes, ...] = tuple(bytes([b]) for b in pre_bytes)
                pre_token_map[pre_tuple] = pre_token_map.get(pre_tuple, 0) + 1

    while len(vocab) < vocab_size:
        pair_count_map.clear()
        for pre_token_bytes, count in pre_token_map.items():
            pre_token_str = b"".join(pre_token_bytes).decode("utf-8")
            if pre_token_str in special_tokens:
                continue
            for a, b in zip(pre_token_bytes, pre_token_bytes[1:]):
                key = (a, b)
                pair_count_map[key] = pair_count_map.get(key, 0) + count
        max_count = 0
        max_pairs = []
        for pair, count in pair_count_map.items():
            if count > max_count:
                max_count = count
                max_pairs.clear()
                max_pairs.append(pair)
            elif count == max_count:
                max_pairs.append(pair)
        max_max_pair = max(max_pairs)
        merges.append(max_max_pair)
        left, right = max_max_pair
        merged_token = left + right

        # add to vocab
        vocab[cur_token_id] = merged_token
        cur_token_id += 1

        new_pre_token_map: dict[tuple[bytes, ...], int] = {}
        special_token_bytes = {s.encode("utf-8") for s in special_tokens}

        for pre_token_bytes, count in pre_token_map.items():
            # never merge inside special tokens
            if b"".join(pre_token_bytes) in special_token_bytes:
                new_pre_token_map[pre_token_bytes] = (
                    new_pre_token_map.get(pre_token_bytes, 0) + count
                )
                continue
            out: list[bytes] = []
            i = 0
            while i < len(pre_token_bytes):
                if i + 1 < len(pre_token_bytes) and pre_token_bytes[i] == left and pre_token_bytes[i + 1] == right:
                    out.append(merged_token)
                    i += 2
                else:
                    out.append(pre_token_bytes[i])
                    i += 1
            out_t = tuple(out)
            new_pre_token_map[out_t] = new_pre_token_map.get(out_t, 0) + count
        pre_token_map = new_pre_token_map


    return (vocab, merges)
