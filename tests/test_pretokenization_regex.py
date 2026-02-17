import regex as re

from cs336_basics.bpe import PRE_TPKEN_PAT


def _pretokenize(text: str) -> list[str]:
    return re.findall(PRE_TPKEN_PAT, text)


def test_pretoken_pattern_words_numbers_punctuation():
    text = "Hello, world! abc123"
    assert _pretokenize(text) == ["Hello", ",", " world", "!", " abc", "123"]


def test_pretoken_pattern_contractions():
    text = "I can't, I've, we're"
    assert _pretokenize(text) == ["I", " can", "'t", ",", " I", "'ve", ",", " we", "'re"]


def test_pretoken_pattern_roundtrip():
    cases = [
        "",
        "Hello world",
        "line1\n\nline2",
        "emoji 🙃 test",
        "  leading",
    ]
    for text in cases:
        assert "".join(_pretokenize(text)) == text
