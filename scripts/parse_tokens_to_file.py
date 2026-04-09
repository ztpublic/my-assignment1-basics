from cs336_basics.tokenizer import Tokenizer
import numpy as np


def main():
    tokenizer = Tokenizer.from_files("data/vocab.json", "data/merges.txt", ["<|endoftext|>"])
    with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
        content = f.read()  
        encoded = tokenizer.encode(content)
        arr = np.ndarray(encoded, dtype=np.int32)
        np.save("data/tiny-stories-10000-tokenized", arr)


if __name__ == "__main__":
    main()
