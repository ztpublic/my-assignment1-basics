from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from functools import lru_cache

from cs336_basics.bpe import PRE_TOKEN_RE, SINGLE_BYTE_TOKENS, get_special_token_re
from cs336_basics.gpt2_utils import gpt2_text_to_bytes


class Tokenizer:
    """Byte-pair-encoding tokenizer with GPT-2-compatible file loading."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # Store the token-id to token-bytes mapping.
        self.vocab = vocab

        # Build the inverse mapping used during encoding.
        self.vocab_inverse = {token_bytes: token_id for token_id, token_bytes in vocab.items()}

        # Store merges in order and precompute their priority rank.
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        # Normalize special-token configuration into immutable text and byte forms.
        special_tokens = special_tokens or []
        self.special_tokens_text = tuple(special_tokens)
        self.special_tokens = {token.encode("utf-8") for token in special_tokens}

        # Compile the special-token regex only if needed.
        self.special_token_re = (
            get_special_token_re(self.special_tokens_text) if self.special_tokens_text else None
        )

        # Build a fast lookup from special-token bytes to their vocab ids.
        self.special_token_dict = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
            if token_bytes in self.special_tokens
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load a tokenizer from GPT-2-style vocab and merges files."""
        # GPT-2 vocab files store printable token text mapped to integer ids.
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            raw_vocab: dict[str, int] = json.load(vocab_file)

        # Convert printable GPT-2 token strings back into their raw byte form.
        vocab = {
            token_id: gpt2_text_to_bytes(token_text)
            for token_text, token_id in raw_vocab.items()
        }

        # Optionally append missing special tokens to the end of the vocabulary.
        specials = special_tokens or []
        if specials:
            vocab_values = set(vocab.values())
            for special_token in specials:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes in vocab_values:
                    continue
                vocab[len(vocab)] = special_token_bytes
                vocab_values.add(special_token_bytes)

        # Parse the merges file line by line.
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                cleaned_line = line.strip()

                # Skip blank lines and header/comment lines such as ``#version: 0.2``.
                if not cleaned_line or cleaned_line.startswith("#"):
                    continue

                # Each merge line should consist of exactly two token strings.
                parts = cleaned_line.split()
                if len(parts) != 2:
                    continue

                left_text, right_text = parts
                merges.append((gpt2_text_to_bytes(left_text), gpt2_text_to_bytes(right_text)))

        return cls(vocab, merges, specials)

    def encode(self, text: str) -> list[int]:
        # Eagerly materialize the iterator-based encoder for convenience.
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an arbitrary iterable of text chunks without concatenating them first."""
        for pretoken_bytes in self._pre_token_iter(iterable):
            # Special tokens bypass BPE merging and map directly to their ids.
            if pretoken_bytes in self.special_tokens:
                yield self.special_token_dict[pretoken_bytes]
                continue

            # Ordinary pretokens are encoded with the merge table.
            yield from self._encode_piece(pretoken_bytes)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a UTF-8 string."""
        # Map ids back to raw token bytes.
        token_bytes = [self.vocab[token_id] for token_id in ids]

        # Concatenate the byte fragments exactly as the tokenizer produced them.
        joined_bytes = b"".join(token_bytes)

        # Decode UTF-8, replacing malformed sequences instead of crashing.
        return joined_bytes.decode("utf-8", errors="replace")

    def _pre_token_iter(self, iterable: Iterable[str]) -> Iterator[bytes]:
        """Yield UTF-8 byte pretokens, preserving special tokens as atomic spans."""
        # The no-special-token path is the simplest and most common case.
        if not self.special_tokens:
            for chunk in iterable:
                for match in PRE_TOKEN_RE.finditer(chunk):
                    yield match.group(0).encode("utf-8")
            return

        # When special tokens exist, split around them and only regex-tokenize the gaps.
        for chunk in iterable:
            last_index = 0

            for match in self.special_token_re.finditer(chunk):
                start_index, end_index = match.span()

                # Pre-tokenize the text before the special token.
                if start_index > last_index:
                    for pretoken_match in PRE_TOKEN_RE.finditer(chunk[last_index:start_index]):
                        yield pretoken_match.group(0).encode("utf-8")

                # Yield the matched special token as a single atomic byte string.
                yield match.group(0).encode("utf-8")
                last_index = end_index

            # Pre-tokenize any trailing text after the final special token.
            if last_index < len(chunk):
                for pretoken_match in PRE_TOKEN_RE.finditer(chunk[last_index:]):
                    yield pretoken_match.group(0).encode("utf-8")

    @lru_cache(maxsize=65536)
    def _encode_piece(self, pre: bytes) -> tuple[int, ...]:
        """Encode one pretoken byte string by repeatedly applying the best merge."""
        # Start from the byte-level vocabulary.
        pieces = [SINGLE_BYTE_TOKENS[byte_value] for byte_value in pre]

        # Greedily apply the highest-priority available merge until no merge remains.
        while len(pieces) > 1:
            best_rank: int | None = None
            best_pair: tuple[bytes, bytes] | None = None

            # Find the adjacent pair with the smallest merge rank.
            for left, right in zip(pieces, pieces[1:]):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # If no pair in the current piece sequence is mergeable, we are done.
            if best_pair is None:
                break

            left, right = best_pair
            merged_piece = left + right
            merged_pieces: list[bytes] = []
            index = 0

            # Rebuild the sequence, merging every matching occurrence of the best pair.
            while index < len(pieces):
                if index + 1 < len(pieces) and pieces[index] == left and pieces[index + 1] == right:
                    merged_pieces.append(merged_piece)
                    index += 2
                else:
                    merged_pieces.append(pieces[index])
                    index += 1

            pieces = merged_pieces

        # Map the final merged byte pieces back into token ids.
        return tuple(self.vocab_inverse[piece] for piece in pieces)
