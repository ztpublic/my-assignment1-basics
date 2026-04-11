from __future__ import annotations

import heapq
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries


PRE_TPKEN_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
PRE_TOKEN_RE = re.compile(PRE_TPKEN_PAT)
SINGLE_BYTE_TOKENS = tuple(bytes([byte_value]) for byte_value in range(256))

PretokenSequence = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


class _ReversePairOrder:
    """Small wrapper that reverses lexicographic ordering inside the heap tie-breaker."""

    __slots__ = ("pair",)

    def __init__(self, pair: Pair) -> None:
        self.pair = pair

    def __lt__(self, other: "_ReversePairOrder") -> bool:
        # ``heapq`` pops the smallest item, so we reverse the pair order to match the
        # assignment's tie-breaking rules.
        return self.pair > other.pair


def _count_chunk_pretokens(chunk_spec: tuple[str, int, int, tuple[str, ...]]) -> dict[bytes, int]:
    """Count pretoken byte strings inside one file chunk."""
    input_path, start, end, special_tokens = chunk_spec
    counts: dict[bytes, int] = {}

    # Read exactly the requested byte span and decode it as UTF-8 text.
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_text = file.read(end - start).decode("utf-8", errors="ignore")

    # If there are no special tokens, the chunk is just regex-pretokenized directly.
    if not special_tokens:
        for match in PRE_TOKEN_RE.finditer(chunk_text):
            pretoken_bytes = match.group(0).encode("utf-8")
            counts[pretoken_bytes] = counts.get(pretoken_bytes, 0) + 1
        return counts

    # Otherwise, split around special tokens and only regex-pretokenize the gaps.
    special_token_re = get_special_token_re(special_tokens)
    last_index = 0

    for match in special_token_re.finditer(chunk_text):
        start_index, end_index = match.span()

        # Count ordinary pretokens before the special token.
        if start_index > last_index:
            for pretoken_match in PRE_TOKEN_RE.finditer(chunk_text[last_index:start_index]):
                pretoken_bytes = pretoken_match.group(0).encode("utf-8")
                counts[pretoken_bytes] = counts.get(pretoken_bytes, 0) + 1

        # Count the special token itself as one atomic token occurrence.
        special_token_bytes = match.group(0).encode("utf-8")
        counts[special_token_bytes] = counts.get(special_token_bytes, 0) + 1
        last_index = end_index

    # Count any remaining trailing text after the last special token.
    if last_index < len(chunk_text):
        for pretoken_match in PRE_TOKEN_RE.finditer(chunk_text[last_index:]):
            pretoken_bytes = pretoken_match.group(0).encode("utf-8")
            counts[pretoken_bytes] = counts.get(pretoken_bytes, 0) + 1

    return counts


@lru_cache(maxsize=32)
def get_special_token_re(special_tokens: tuple[str, ...]) -> re.Pattern:
    """Compile a regex that matches the provided special tokens longest-first."""
    escaped_special_tokens = "|".join(
        re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
    )
    return re.compile(f"(?:{escaped_special_tokens})")


def _pair_occurrences(pre_token_bytes: PretokenSequence) -> dict[Pair, int]:
    """Count adjacent pair occurrences inside one unique pretoken sequence."""
    pair_occurrence_map: dict[Pair, int] = {}
    for left, right in zip(pre_token_bytes, pre_token_bytes[1:]):
        pair = (left, right)
        pair_occurrence_map[pair] = pair_occurrence_map.get(pair, 0) + 1
    return pair_occurrence_map


def _merge_pair_in_sequence(
    pre_token_bytes: PretokenSequence,
    pair: Pair,
    merged_token: bytes,
) -> PretokenSequence:
    """Merge every left-to-right occurrence of ``pair`` inside one sequence."""
    left, right = pair
    merged_sequence: list[bytes] = []
    index = 0

    while index < len(pre_token_bytes):
        if (
            index + 1 < len(pre_token_bytes)
            and pre_token_bytes[index] == left
            and pre_token_bytes[index + 1] == right
        ):
            merged_sequence.append(merged_token)
            index += 2
        else:
            merged_sequence.append(pre_token_bytes[index])
            index += 1

    return tuple(merged_sequence)


def _push_pair_heap_entry(pair_heap: list[tuple[int, _ReversePairOrder, Pair]], pair: Pair, count: int) -> None:
    """Push one pair candidate into the max-heap encoded via negated counts."""
    heapq.heappush(pair_heap, (-count, _ReversePairOrder(pair), pair))


def _pop_best_pair(
    pair_heap: list[tuple[int, _ReversePairOrder, Pair]],
    pair_counts: dict[Pair, int],
) -> Pair | None:
    """Pop the best currently valid pair from the heap."""
    while pair_heap:
        negated_count, _, pair = heapq.heappop(pair_heap)
        current_count = pair_counts.get(pair, 0)

        # Skip deleted entries.
        if current_count <= 0:
            continue

        # Skip stale heap entries whose stored count is outdated.
        if -negated_count != current_count:
            continue

        return pair

    return None


def my_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    kwargs: dict | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-pair-encoding tokenizer from a text corpus."""
    kwargs = kwargs or {}

    # The base vocabulary always starts with the 256 byte values.
    vocab: dict[int, bytes] = {index: SINGLE_BYTE_TOKENS[index] for index in range(256)}

    # Add special tokens immediately after the byte vocabulary.
    next_token_id = 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    # If the requested vocab is smaller than the mandatory starting vocab, stop early.
    if vocab_size < len(vocab):
        raise ValueError("vocab_size must be at least 256 + len(special_tokens)")

    # ``merges`` records the merge history in creation order.
    merges: list[Pair] = []

    # ``pre_token_map`` stores unique pretoken sequences and their corpus counts.
    pre_token_map: dict[PretokenSequence, int] = {}

    # Choose the worker count, capped to avoid oversubscribing small machines.
    num_processes = max(1, int(kwargs.get("num_processes", min(8, os.cpu_count() or 1))))
    input_path_str = os.fspath(input_path)

    # Split the file into chunks that do not cut through the designated boundary token.
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")

    special_tokens_tuple = tuple(special_tokens)
    chunk_specs = [
        (input_path_str, start, end, special_tokens_tuple)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    # Aggregate raw pretoken byte-string counts across chunks.
    pretoken_byte_counts: dict[bytes, int] = {}
    if num_processes > 1 and len(chunk_specs) > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            chunk_results = executor.map(_count_chunk_pretokens, chunk_specs)
            for chunk_counts in chunk_results:
                for pretoken_bytes, count in chunk_counts.items():
                    pretoken_byte_counts[pretoken_bytes] = pretoken_byte_counts.get(pretoken_bytes, 0) + count
    else:
        for chunk_spec in chunk_specs:
            chunk_counts = _count_chunk_pretokens(chunk_spec)
            for pretoken_bytes, count in chunk_counts.items():
                pretoken_byte_counts[pretoken_bytes] = pretoken_byte_counts.get(pretoken_bytes, 0) + count

    # Convert each byte string into a tuple of byte-level tokens.
    for pretoken_bytes, count in pretoken_byte_counts.items():
        pretoken_sequence = tuple(SINGLE_BYTE_TOKENS[byte_value] for byte_value in pretoken_bytes)
        pre_token_map[pretoken_sequence] = count

    # Special-token sequences never participate in merges.
    special_token_sequences = {
        tuple(SINGLE_BYTE_TOKENS[byte_value] for byte_value in special_token.encode("utf-8"))
        for special_token in special_tokens
    }

    # ``pair_counts`` stores the corpus-wide weighted frequency of each adjacent pair.
    pair_counts: dict[Pair, int] = {}

    # ``pair_to_pre_tokens`` stores the reverse index: which unique pretokens contain a pair.
    pair_to_pre_tokens: dict[Pair, set[PretokenSequence]] = {}

    # ``pre_token_pair_occurrences`` stores pair multiplicities within each unique pretoken.
    pre_token_pair_occurrences: dict[PretokenSequence, dict[Pair, int]] = {}

    # The heap lets us extract the best pair without scanning every pair each merge step.
    pair_heap: list[tuple[int, _ReversePairOrder, Pair]] = []

    # Initialize the pair-count data structures from the starting pretoken inventory.
    for pretoken_sequence, pretoken_count in pre_token_map.items():
        if pretoken_sequence in special_token_sequences or len(pretoken_sequence) < 2:
            pre_token_pair_occurrences[pretoken_sequence] = {}
            continue

        occurrence_map = _pair_occurrences(pretoken_sequence)
        pre_token_pair_occurrences[pretoken_sequence] = occurrence_map

        for pair, pair_occurrence_count in occurrence_map.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + pair_occurrence_count * pretoken_count
            pair_to_pre_tokens.setdefault(pair, set()).add(pretoken_sequence)

    # Seed the heap with the initial pair frequencies.
    for pair, count in pair_counts.items():
        _push_pair_heap_entry(pair_heap, pair, count)

    while len(vocab) < vocab_size:
        # Select the most frequent currently valid pair.
        best_pair = _pop_best_pair(pair_heap, pair_counts)
        if best_pair is None:
            break

        # Record the merge in order.
        merges.append(best_pair)

        # The merged token bytes are just the concatenation of the pair bytes.
        merged_token = best_pair[0] + best_pair[1]

        # Add the new merged token to the vocabulary.
        vocab[next_token_id] = merged_token
        next_token_id += 1

        # Only pretokens containing the selected pair need to be updated.
        affected_pre_tokens = list(pair_to_pre_tokens.get(best_pair, ()))
        merged_pretoken_deltas: dict[PretokenSequence, int] = {}
        changed_pairs: set[Pair] = set()

        # Remove the old pair-count contributions from every affected pretoken.
        for pretoken_sequence in affected_pre_tokens:
            pretoken_count = pre_token_map.pop(pretoken_sequence, 0)
            if pretoken_count <= 0:
                continue

            old_occurrence_map = pre_token_pair_occurrences.pop(pretoken_sequence, {})
            for pair, pair_occurrence_count in old_occurrence_map.items():
                updated_count = pair_counts[pair] - pair_occurrence_count * pretoken_count
                if updated_count > 0:
                    pair_counts[pair] = updated_count
                else:
                    pair_counts.pop(pair, None)
                changed_pairs.add(pair)

                pretokens_for_pair = pair_to_pre_tokens.get(pair)
                if pretokens_for_pair is not None:
                    pretokens_for_pair.discard(pretoken_sequence)
                    if not pretokens_for_pair:
                        pair_to_pre_tokens.pop(pair, None)

            # Apply the selected merge to the pretoken sequence itself.
            merged_pretoken_sequence = _merge_pair_in_sequence(pretoken_sequence, best_pair, merged_token)
            merged_pretoken_deltas[merged_pretoken_sequence] = (
                merged_pretoken_deltas.get(merged_pretoken_sequence, 0) + pretoken_count
            )

        # Add the updated pretokens back into the corpus inventory.
        for pretoken_sequence, delta_count in merged_pretoken_deltas.items():
            pre_token_map[pretoken_sequence] = pre_token_map.get(pretoken_sequence, 0) + delta_count

            # Compute pair occurrences for this unique pretoken once and cache them.
            if pretoken_sequence not in pre_token_pair_occurrences:
                if pretoken_sequence in special_token_sequences or len(pretoken_sequence) < 2:
                    pre_token_pair_occurrences[pretoken_sequence] = {}
                else:
                    pre_token_pair_occurrences[pretoken_sequence] = _pair_occurrences(pretoken_sequence)

            occurrence_map = pre_token_pair_occurrences[pretoken_sequence]
            for pair, pair_occurrence_count in occurrence_map.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + pair_occurrence_count * delta_count
                pair_to_pre_tokens.setdefault(pair, set()).add(pretoken_sequence)
                changed_pairs.add(pair)

        # Refresh heap entries only for pairs whose counts actually changed.
        for pair in changed_pairs:
            updated_count = pair_counts.get(pair, 0)
            if updated_count > 0:
                _push_pair_heap_entry(pair_heap, pair, updated_count)

    return vocab, merges
