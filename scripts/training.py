from cs336_basics.transformer import TransformerLMConfig


def main():
    config = TransformerLMConfig(
        vocab_size=10000,
        context_length=1_024,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
        d_ff=6_400,
    )

    pass

if __name__ == "__main__":
    main()
