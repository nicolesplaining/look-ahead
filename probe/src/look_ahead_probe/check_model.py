"""Check model properties before running activation extraction."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_model(model_name: str, device: str = "cuda", **kwargs):
    """Print model metadata and confirm it's loadable."""
    print(f"{'='*60}")
    print(f"MODEL CHECK: {model_name}")
    print(f"{'='*60}\n")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cfg = model.config
    print(f"\n{'='*60}")
    print("MODEL METADATA")
    print(f"{'='*60}")
    print(f"Model name:        {model_name}")
    print(f"Number of layers:  {cfg.num_hidden_layers}")
    print(f"Hidden dimension:  {cfg.hidden_size}")
    print(f"Vocabulary size:   {cfg.vocab_size}")
    print(f"Attention heads:   {cfg.num_attention_heads}")
    if hasattr(cfg, 'max_position_embeddings'):
        print(f"Context length:    {cfg.max_position_embeddings}")
    if hasattr(cfg, 'intermediate_size'):
        print(f"MLP dimension:     {cfg.intermediate_size}")

    print(f"\nDevice:            {device}")
    print(f"Model dtype:       {next(model.parameters()).dtype}")

    print(f"\n{'='*60}")
    print("PROBING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Available layers for probing: 0 to {cfg.num_hidden_layers - 1}")
    print(f"Activation dimension (d_model): {cfg.hidden_size}")
    print(f"Output vocabulary size: {cfg.vocab_size}")

    print(f"\n✓ Model loaded successfully.")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Check model properties")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Unused; kept for CLI compatibility with layer_k_experiment.py")

    args = parser.parse_args()
    check_model(model_name=args.model_name, device=args.device)


if __name__ == "__main__":
    main()
