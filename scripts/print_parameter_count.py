# vocab_size : 50,257
# context_length : 1,024
# num_layers : 48
# d_model : 1,600
# num_heads : 25
# d_ff : 6,400
from cs336_basics.transformer import TransformerLM, TransformerLMConfig


def main():
    config = TransformerLMConfig(
        vocab_size=50_257,
        context_length=1_024,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
        d_ff=6_400,
    )
    model = TransformerLM.from_config(config, device="meta")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"{name}: {parameter.numel():,}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_memory_mb = parameter_count * 4 / (1024 ** 2)
    print(f"TransformerLM parameter count: {parameter_count:,}")
    print(f"TransformerLM fp32 parameter memory: {total_memory_mb:,.2f} MB")

if __name__ == "__main__":
    main()
