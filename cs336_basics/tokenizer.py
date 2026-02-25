from collections.abc import Iterable, Iterator
import json

from cs336_basics.bpe import PRE_TOKEN_RE, get_special_token_re
from cs336_basics.gpt2_utils import gpt2_text_to_bytes


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.vocab_inverse: dict[bytes, int] = {}
        for k,v in vocab.items():
            self.vocab_inverse[v] = k
        self.merges = merges
        if special_tokens == None:
            special_tokens = []
        self.special_tokens = set([s.encode("utf-8") for s in special_tokens])
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
            pre_bytes = tuple(bytes([i]) for i in pre)
            yield from self._encode_pre_token(pre_bytes)

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
        special_token_re = get_special_token_re(tuple(s.decode("utf-8") for s in self.special_tokens))
        for chunk in iterable:
            last_index = 0
            for match in special_token_re.finditer(chunk):
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

    def _encode_pre_token(self, pre_bytes: tuple[bytes, ...]) -> Iterator[int]:
        matched = True
        while True:
            if not matched:
                break
            matched = False
            pair_map = self._get_adjacent_pair_map(pre_bytes)
            for merge in self.merges:
                if merge in pair_map:
                    matched = True
                    merge_start_idx = pair_map[merge]
                    a, b = merge
                    merged_bytes = a + b
                    new_pre_bytes = (*pre_bytes[:merge_start_idx], merged_bytes, *pre_bytes[merge_start_idx + 2:])
                    pre_bytes = new_pre_bytes
                    break
        for b in pre_bytes:
            yield self.vocab_inverse[b]            


    def _get_adjacent_pair_map(self, pre_bytes: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
        out = {}
        for idx, pre in enumerate(pre_bytes):
            if idx < len(pre_bytes) - 1:
                bytes_tuple = (pre, pre_bytes[idx + 1])
                if bytes_tuple not in out:
                    out[bytes_tuple] = idx
        return out
