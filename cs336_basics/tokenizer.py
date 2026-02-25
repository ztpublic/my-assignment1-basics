from collections.abc import Iterable, Iterator

from cs336_basics.bpe import PRE_TOKEN_RE, get_special_token_re


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.vocab_inverse: dict[bytes, int] = {}
        for k,v in vocab.items():
            self.vocab_inverse[v] = k
        self.merges = merges
        self.special_tokens = set([s.encode("utf-8") for s in special_tokens])
        self.special_token_dict: dict[bytes, int] = {}
        for idx, b in vocab.items():
            if b in self.special_tokens:
                self.special_token_dict[b] = idx
            
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        pass 

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for pre in self._pre_token_iter(iterable):
            if pre in self.special_tokens:
                yield self.special_token_dict[pre]
                continue
            pre_bytes = tuple(bytes([i]) for i in pre)


    def decode(self, ids: list[int]) -> str:
        pass

    def _pre_token_iter(self, iterable: Iterable[str]) -> Iterator[bytes]:
        if not self.special_tokens:
            for chunk in iterable:
                for pre in PRE_TOKEN_RE.finditer(chunk):
                   pre_bytes = pre.group(0).encode("utf-8")
                   yield pre_bytes
            return
        special_token_re = get_special_token_re(tuple(self.special_tokens))
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

    def _merge_pre_token_bytes(self, pre_bytes: tuple[bytes]) -> Iterator[int]:
        pass