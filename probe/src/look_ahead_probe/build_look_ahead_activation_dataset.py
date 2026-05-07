"""Build look-ahead activation dataset for probe training."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .activation_extraction import generate_and_extract_all_layers, merge_layer_chunks, get_n_layers, get_hidden_size, get_vocab_size
from .data_loading import load_jsonl_prompts


def main():
    parser = argparse.ArgumentParser(description="Build multi-layer activation dataset")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--prompts_path", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--split_field", type=str, default="split")
    parser.add_argument("--split_value", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)

    parser.add_argument("--max_k", type=int, required=True,
                        help="Maximum lookahead distance (extracts targets for k=1,2,...,max_k)")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices or None for all layers")

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["4bit", "8bit"],
                        help="Quantize the model: '8bit' halves bfloat16 memory, "
                             "'4bit' quarters it (requires bitsandbytes)")

    args = parser.parse_args()

    bnb_config = None
    if args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    print(f"Loading model: {args.model_name}"
          + (f" [{args.quantization} quantization]" if args.quantization else ""))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **({"quantization_config": bnb_config} if bnb_config else {"torch_dtype": torch.bfloat16}),
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / '.chunks'

    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(',')]
    else:
        layers = None

    print("Loading prompts...")
    if args.prompts_path is not None:
        prompts = load_jsonl_prompts(
            args.prompts_path, text_field=args.text_field,
            split_field=None, max_examples=args.max_prompts
        )
    elif args.dataset_path is not None:
        prompts = load_jsonl_prompts(
            args.dataset_path, text_field=args.text_field,
            split_field=args.split_field, split_value=args.split_value,
            max_examples=args.max_prompts
        )
    else:
        raise ValueError("Must provide --prompts_path or --dataset_path")

    print(f"Loaded {len(prompts)} prompts")

    resolved_layers = layers if layers is not None else list(range(get_n_layers(model)))
    metadata = {
        'model_name': args.model_name,
        'max_k': args.max_k,
        'max_new_tokens': args.max_new_tokens,
        'n_prompts': len(prompts),
        'd_model': get_hidden_size(model),
        'vocab_size': get_vocab_size(model),
        'layers': resolved_layers,
    }

    print(f"Extracting activations (max_k={args.max_k})...")
    _, targets, generated_texts, generated_token_ids = generate_and_extract_all_layers(
        model=model, tokenizer=tokenizer, prompts=prompts, max_k=args.max_k,
        max_new_tokens=args.max_new_tokens, device=args.device, layers=layers,
        chunk_dir=str(chunk_dir),
    )

    print(f"\nExample texts (first 3):")
    for i, text in enumerate(generated_texts[:3]):
        print(f"  {i+1}. {text[:100]}...")

    print(f"\nMerging layer chunks (one layer at a time)...")
    merge_layer_chunks(str(chunk_dir), str(output_dir), resolved_layers)

    torch.save(targets, output_dir / 'targets.pt')
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Sidecar files saved as siblings to the output directory.
    # Baselines use .tokens.jsonl directly — no decode/re-encode roundtrip.
    tokens_path = output_dir.parent / (output_dir.name + '.tokens.jsonl')
    with open(tokens_path, 'w', encoding='utf-8') as f:
        for token_ids in generated_token_ids:
            f.write(json.dumps({'tokens': token_ids}) + '\n')

    texts_path = output_dir.parent / (output_dir.name + '.texts.jsonl')
    with open(texts_path, 'w', encoding='utf-8') as f:
        for text in generated_texts:
            f.write(json.dumps({'text': text}) + '\n')

    print(f"\nDataset saved to:   {output_dir}/")
    print(f"Token IDs saved to: {tokens_path}  ({tokens_path.stat().st_size // 1024} KB)")
    print(f"Texts saved to:     {texts_path}  ({texts_path.stat().st_size // 1024} KB)")
    print(f"Layers: {resolved_layers}")
    print(f"Samples: {len(targets)}, Targets shape: {targets.shape}")


if __name__ == "__main__":
    main()
