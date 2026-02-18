import os
import heapq
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re


PRE_TPKEN_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
PRE_TOKEN_RE = re.compile(PRE_TPKEN_PAT)
SINGLE_BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


class _ReversePairOrder:
    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_ReversePairOrder") -> bool:
        return self.pair > other.pair


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


def _pair_occurrences(
    pre_token_bytes: tuple[bytes, ...],
) -> dict[tuple[bytes, bytes], int]:
    pair_occurrence_map: dict[tuple[bytes, bytes], int] = {}
    for left, right in zip(pre_token_bytes, pre_token_bytes[1:]):
        pair = (left, right)
        pair_occurrence_map[pair] = pair_occurrence_map.get(pair, 0) + 1
    return pair_occurrence_map


def _merge_pair_in_sequence(
    pre_token_bytes: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    left, right = pair
    out: list[bytes] = []
    i = 0
    while i < len(pre_token_bytes):
        if i + 1 < len(pre_token_bytes) and pre_token_bytes[i] == left and pre_token_bytes[i + 1] == right:
            out.append(merged_token)
            i += 2
        else:
            out.append(pre_token_bytes[i])
            i += 1
    return tuple(out)


def _push_pair_heap_entry(
    pair_heap: list[tuple[int, _ReversePairOrder, tuple[bytes, bytes]]],
    pair: tuple[bytes, bytes],
    count: int,
) -> None:
    heapq.heappush(pair_heap, (-count, _ReversePairOrder(pair), pair))


def _pop_best_pair(
    pair_heap: list[tuple[int, _ReversePairOrder, tuple[bytes, bytes]]],
    pair_counts: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes] | None:
    while pair_heap:
        neg_count, _, pair = heapq.heappop(pair_heap)
        current_count = pair_counts.get(pair, 0)
        if current_count <= 0:
            continue
        if -neg_count != current_count:
            continue
        return pair
    return None


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

    # pre-token -> frequency
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

    special_token_tuples = {
        tuple(SINGLE_BYTE_TOKENS[b] for b in s.encode("utf-8"))
        for s in special_tokens
    }

    # Incremental merge state:
    # - pair_counts: global weighted pair frequency across all pre-tokens
    # - pair_to_pre_tokens: reverse index of which pre-tokens contain each pair
    # - pre_token_pair_occurrences: pair multiplicities inside each unique pre-token
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_pre_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    pre_token_pair_occurrences: dict[tuple[bytes, ...], dict[tuple[bytes, bytes], int]] = {}
    pair_heap: list[tuple[int, _ReversePairOrder, tuple[bytes, bytes]]] = []

    for pre_token_bytes, pre_token_count in pre_token_map.items():
        if pre_token_bytes in special_token_tuples or len(pre_token_bytes) < 2:
            pre_token_pair_occurrences[pre_token_bytes] = {}
            continue
        occurrence_map = _pair_occurrences(pre_token_bytes)
        pre_token_pair_occurrences[pre_token_bytes] = occurrence_map
        for pair, pair_occurrence_count in occurrence_map.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + (pair_occurrence_count * pre_token_count)
            pair_to_pre_tokens.setdefault(pair, set()).add(pre_token_bytes)

    for pair, count in pair_counts.items():
        _push_pair_heap_entry(pair_heap, pair, count)

    while len(vocab) < vocab_size:
        max_pair = _pop_best_pair(pair_heap, pair_counts)
        if max_pair is None:
            break

        merges.append(max_pair)
        merged_token = max_pair[0] + max_pair[1]

        # add to vocab
        vocab[cur_token_id] = merged_token
        cur_token_id += 1

        affected_pre_tokens = list(pair_to_pre_tokens.get(max_pair, ()))
        merged_pre_token_deltas: dict[tuple[bytes, ...], int] = {}
        changed_pairs: set[tuple[bytes, bytes]] = set()

        # Remove old contributions for pre-tokens that contain the selected pair.
        for pre_token_bytes in affected_pre_tokens:
            pre_token_count = pre_token_map.pop(pre_token_bytes, 0)
            if pre_token_count <= 0:
                continue

            old_occurrence_map = pre_token_pair_occurrences.pop(pre_token_bytes, {})
            for pair, pair_occurrence_count in old_occurrence_map.items():
                updated_count = pair_counts[pair] - (pair_occurrence_count * pre_token_count)
                if updated_count > 0:
                    pair_counts[pair] = updated_count
                else:
                    pair_counts.pop(pair, None)
                changed_pairs.add(pair)

                pre_tokens_for_pair = pair_to_pre_tokens.get(pair)
                if pre_tokens_for_pair is not None:
                    pre_tokens_for_pair.discard(pre_token_bytes)
                    if not pre_tokens_for_pair:
                        pair_to_pre_tokens.pop(pair, None)

            merged_pre_token = _merge_pair_in_sequence(pre_token_bytes, max_pair, merged_token)
            merged_pre_token_deltas[merged_pre_token] = (
                merged_pre_token_deltas.get(merged_pre_token, 0) + pre_token_count
            )

        # Add contributions for updated pre-tokens after applying the selected merge.
        for pre_token_bytes, delta_count in merged_pre_token_deltas.items():
            previous_count = pre_token_map.get(pre_token_bytes, 0)
            pre_token_map[pre_token_bytes] = previous_count + delta_count

            if pre_token_bytes not in pre_token_pair_occurrences:
                if pre_token_bytes in special_token_tuples or len(pre_token_bytes) < 2:
                    pre_token_pair_occurrences[pre_token_bytes] = {}
                else:
                    pre_token_pair_occurrences[pre_token_bytes] = _pair_occurrences(pre_token_bytes)

            occurrence_map = pre_token_pair_occurrences[pre_token_bytes]
            for pair, pair_occurrence_count in occurrence_map.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + (pair_occurrence_count * delta_count)
                pair_to_pre_tokens.setdefault(pair, set()).add(pre_token_bytes)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            updated_count = pair_counts.get(pair, 0)
            if updated_count > 0:
                _push_pair_heap_entry(pair_heap, pair, updated_count)


    return (vocab, merges)
