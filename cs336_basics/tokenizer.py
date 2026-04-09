from collections.abc import Iterable, Iterator
from functools import lru_cache
import json

from cs336_basics.bpe import PRE_TOKEN_RE, SINGLE_BYTE_TOKENS, get_special_token_re
from cs336_basics.gpt2_utils import gpt2_text_to_bytes


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.vocab_inverse: dict[bytes, int] = {}
        for k,v in vocab.items():
            self.vocab_inverse[v] = k
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        if special_tokens == None:
            special_tokens = []
        self.special_tokens = set([s.encode("utf-8") for s in special_tokens])
        self.special_tokens_text = tuple(special_tokens)
        self.special_token_re = (
            get_special_token_re(self.special_tokens_text)
            if self.special_tokens_text
            else None
        )
        self.special_token_dict: dict[bytes, int] = {}
        for idx, b in vocab.items():
            if b in self.special_tokens:
                self.special_token_dict[b] = idx
            
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            gpt2_vocab: dict[str, int] = json.load(vocab_f)

        vocab = {
            token_id: gpt2_text_to_bytes(token_text)
            for token_text, token_id in gpt2_vocab.items()
        }

        specials = special_tokens or []
        if specials:
            vocab_values = set(vocab.values())
            for special_token in specials:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in vocab_values:
                    vocab[len(vocab)] = special_token_bytes
                    vocab_values.add(special_token_bytes)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_f:
            for line in merges_f:
                cleaned_line = line.rstrip()
                if not cleaned_line:
                    continue
                parts = cleaned_line.split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                left_bytes = gpt2_text_to_bytes(left)
                right_bytes = gpt2_text_to_bytes(right)
                merges.append((left_bytes, right_bytes))

        return cls(vocab, merges, specials)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for pre in self._pre_token_iter(iterable):
            if pre in self.special_tokens:
                yield self.special_token_dict[pre]
                continue
            yield from self._encode_piece(pre)

    def decode(self, ids: list[int]) -> str:
        byte_list = [self.vocab[token_id] for token_id in ids]
        joined = b"".join(byte_list)
        return joined.decode("utf-8", errors="replace")

    def _pre_token_iter(self, iterable: Iterable[str]) -> Iterator[bytes]:
        if not self.special_tokens:
            for chunk in iterable:
                for pre in PRE_TOKEN_RE.finditer(chunk):
                   pre_bytes = pre.group(0).encode("utf-8")
                   yield pre_bytes
            return
        for chunk in iterable:
            last_index = 0
            for match in self.special_token_re.finditer(chunk):
                start_index, end_index = match.span()
                if start_index > last_index:
                    for pre in PRE_TOKEN_RE.finditer(chunk[last_index:start_index]):
                        pre_bytes = pre.group(0).encode("utf-8")    
                        yield pre_bytes   
                special = match.group(0)
                yield special.encode("utf-8")
                last_index = end_index
            if last_index < len(chunk):
                for pre in PRE_TOKEN_RE.finditer(chunk[last_index:]):
                    pre_bytes = pre.group(0).encode("utf-8")
                    yield pre_bytes

    @lru_cache(maxsize=65536)
    def _encode_piece(self, pre: bytes) -> tuple[int, ...]:
        pieces = [SINGLE_BYTE_TOKENS[b] for b in pre]
        while len(pieces) > 1:
            best_rank: int | None = None
            best_pair: tuple[bytes, bytes] | None = None

            for left, right in zip(pieces, pieces[1:]):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            left, right = best_pair
            merged = left + right
            new_pieces: list[bytes] = []
            idx = 0
            while idx < len(pieces):
                if idx + 1 < len(pieces) and pieces[idx] == left and pieces[idx + 1] == right:
                    new_pieces.append(merged)
                    idx += 2
                else:
                    new_pieces.append(pieces[idx])
                    idx += 1
            pieces = new_pieces

        return tuple(self.vocab_inverse[piece] for piece in pieces)
