from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte-to-text mapping used by GPT-2 BPE files."""
    # GPT-2 keeps common printable bytes as themselves.
    visible_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    # The Unicode code points start as an identity mapping for visible bytes.
    unicode_codepoints = visible_bytes[:]

    # Bytes outside the visible set are remapped into a private extended range.
    extra_codepoint_offset = 0
    for byte_value in range(256):
        if byte_value in visible_bytes:
            continue
        visible_bytes.append(byte_value)
        unicode_codepoints.append(256 + extra_codepoint_offset)
        extra_codepoint_offset += 1

    # Zip the final byte list to the printable surrogate characters.
    return dict(zip(visible_bytes, (chr(codepoint) for codepoint in unicode_codepoints)))


@lru_cache(maxsize=1)
def gpt2_unicode_to_bytes() -> dict[str, int]:
    """Return the inverse GPT-2 text-to-byte mapping."""
    return {text_char: byte_value for byte_value, text_char in gpt2_bytes_to_unicode().items()}


def gpt2_text_to_bytes(token_text: str) -> bytes:
    """Decode a GPT-2 BPE token string back into its raw bytes."""
    decoder = gpt2_unicode_to_bytes()
    return bytes(decoder[character] for character in token_text)


def bytes_to_gpt2_text(token_bytes: bytes) -> str:
    """Encode raw bytes into the GPT-2 printable token representation."""
    encoder = gpt2_bytes_to_unicode()
    return "".join(encoder[byte_value] for byte_value in token_bytes)
