"""Check model properties and verify efficient activation extraction works."""

import argparse
import torch
from transformer_lens import HookedTransformer
from .activation_extraction import verify_activation_equivalence


def check_model(model_name: str, device: str = "cuda", test_layer: int = None, max_new_tokens: int = 20):
    """
    Inspect model internals and verify activation extraction assumptions.

    Args:
        model_name: Name of model to check
        device: Device to use
        test_layer: Layer to test (default: middle layer)
        max_new_tokens: Tokens to generate for verification test
    """
    print(f"{'='*60}")
    print(f"MODEL CHECK: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    # Print model metadata
    print(f"\n{'='*60}")
    print("MODEL METADATA")
    print(f"{'='*60}")
    print(f"Model name:        {model_name}")
    print(f"Number of layers:  {model.cfg.n_layers}")
    print(f"Hidden dimension:  {model.cfg.d_model}")
    print(f"Vocabulary size:   {model.cfg.d_vocab}")
    print(f"Attention heads:   {model.cfg.n_heads}")
    print(f"Context length:    {model.cfg.n_ctx}")

    if hasattr(model.cfg, 'd_head'):
        print(f"Head dimension:    {model.cfg.d_head}")
    if hasattr(model.cfg, 'd_mlp'):
        print(f"MLP dimension:     {model.cfg.d_mlp}")

    print(f"\nDevice:            {device}")
    print(f"Model dtype:       {next(model.parameters()).dtype}")

    # Probing-relevant info
    print(f"\n{'='*60}")
    print("PROBING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Available layers for probing: 0 to {model.cfg.n_layers - 1}")
    print(f"Activation dimension (d_model): {model.cfg.d_model}")
    print(f"Output vocabulary size: {model.cfg.d_vocab}")

    # Verify activation equivalence
    if test_layer is None:
        test_layer = model.cfg.n_layers // 2  # Middle layer

    print(f"\n{'='*60}")
    print("ACTIVATION EXTRACTION VERIFICATION")
    print(f"{'='*60}")
    print(f"Testing layer: {test_layer}")
    print(f"Max new tokens: {max_new_tokens}")
    print("\nThis checks that the i-th activation depends only on tokens 0..i")
    print("(validating the causal masking property)\n")

    test_prompt = "The quick brown fox jumps over the lazy dog."
    is_valid = verify_activation_equivalence(
        model=model,
        prompt=test_prompt,
        layer_idx=test_layer,
        max_new_tokens=max_new_tokens,
        device=device
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}, vocab: {model.cfg.d_vocab}")

    if is_valid:
        print(f"\n✓ VERIFICATION PASSED")
        print(f"  Efficient activation extraction is safe for this model.")
        print(f"  You can use build_look_ahead_activation_dataset.py")
    else:
        print(f"\n✗ VERIFICATION FAILED")
        print(f"  Activations differ between generation and single-pass extraction.")
        print(f"  This model may not be suitable for efficient extraction.")
        print(f"  Consider using token-by-token extraction instead.")

    print(f"{'='*60}\n")

    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Check model properties and verify extraction")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Model to check (e.g., gpt2-small)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--test_layer", type=int, default=None,
                        help="Layer to test (default: middle layer)")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Tokens to generate for verification")

    args = parser.parse_args()

    check_model(
        model_name=args.model_name,
        device=args.device,
        test_layer=args.test_layer,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()

