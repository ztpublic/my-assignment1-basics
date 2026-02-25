from functools import lru_cache


@lru_cache(maxsize=1)
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Map each byte (0..255) to a printable GPT-2 unicode surrogate."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(x) for x in cs)))


@lru_cache(maxsize=1)
def gpt2_unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


def gpt2_text_to_bytes(token_text: str) -> bytes:
    decoder = gpt2_unicode_to_bytes()
    return bytes(decoder[ch] for ch in token_text)


def bytes_to_gpt2_text(token_bytes: bytes) -> str:
    encoder = gpt2_bytes_to_unicode()
    return "".join(encoder[b] for b in token_bytes)
