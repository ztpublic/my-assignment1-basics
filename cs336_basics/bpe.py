import os
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re


PRE_TPKEN_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
PRE_TOKEN_RE = re.compile(PRE_TPKEN_PAT)
SINGLE_BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


def _count_chunk_pretokens(
    chunk_spec: tuple[str, int, int],
) -> dict[bytes, int]:
    input_path, start, end = chunk_spec
    counts: dict[bytes, int] = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    for pre in PRE_TOKEN_RE.finditer(chunk):
        pre_bytes = pre.group(0).encode("utf-8")
        counts[pre_bytes] = counts.get(pre_bytes, 0) + 1

    return counts


def my_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    kwargs: dict | None = None,
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
    kwargs = kwargs or {}

    vocab: dict[int, bytes] = {i: SINGLE_BYTE_TOKENS[i] for i in range(256)}

    cur_token_id = 256

    for special in special_tokens:
        vocab[cur_token_id] = special.encode("utf-8")
        cur_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

    pair_count_map: dict[tuple[bytes, bytes], int] = {}

    pre_token_map: dict[tuple[bytes, ...], int] = {}

    # initial pair count for pre-tokens
    num_processes = max(1, int(kwargs.get("num_processes", min(8, os.cpu_count() or 1))))
    input_path_str = os.fspath(input_path)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    chunk_specs = [
        (input_path_str, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    pre_token_bytes_counts: dict[bytes, int] = {}
    # Always process chunks via multiprocessing.
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_counts in executor.map(_count_chunk_pretokens, chunk_specs):
            for pre_bytes, count in chunk_counts.items():
                pre_token_bytes_counts[pre_bytes] = (
                    pre_token_bytes_counts.get(pre_bytes, 0) + count
                )

    for pre_bytes, count in pre_token_bytes_counts.items():
        pre_tuple = tuple(SINGLE_BYTE_TOKENS[b] for b in pre_bytes)
        pre_token_map[pre_tuple] = count

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
