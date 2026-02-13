#!/usr/bin/env python3
"""
N-gram baseline models for the look-ahead probe experiment.

Trains unigram, bigram, and trigram models on generated texts from the
training activation dataset and evaluates them on the validation set.

Uses skip-k n-grams so the context is always the token(s) at position i
(same "view" as the probe activation), predicting the token at position i+k.

No smoothing: unseen contexts score 0 (conservative baseline).

Outputs one JSON per baseline model in the same schema as
experiment_results.json, with accuracy replicated across all layer indices
so results appear as flat horizontal lines when fed into plot_results.sh.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# N-gram training
# ---------------------------------------------------------------------------

def build_ngram_tables(
    token_sequences: List[List[int]],
    max_k: int,
) -> Tuple[Counter, Dict, Dict]:
    """
    Build skip-k unigram, bigram, and trigram frequency tables for each k.

    For each k in 1..max_k:
      - bigram_k[k][ctx]        = Counter {token: count} for P(t_{i+k} | t_i)
      - trigram_k[k][(p, ctx)]  = Counter {token: count} for P(t_{i+k} | t_{i-1}, t_i)

    Unigram is k-independent: just overall token frequency.
    """
    unigram: Counter = Counter()
    bigram_k: Dict[int, Dict[int, Counter]] = {k: defaultdict(Counter) for k in range(1, max_k + 1)}
    trigram_k: Dict[int, Dict[Tuple, Counter]] = {k: defaultdict(Counter) for k in range(1, max_k + 1)}

    for tokens in token_sequences:
        n = len(tokens)
        for i, tok in enumerate(tokens):
            unigram[tok] += 1

        for k in range(1, max_k + 1):
            for i in range(n - k):
                target = tokens[i + k]
                ctx = tokens[i]
                bigram_k[k][ctx][target] += 1

                if i >= 1:
                    prev_ctx = tokens[i - 1]
                    trigram_k[k][(prev_ctx, ctx)][target] += 1

    return unigram, bigram_k, trigram_k


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def top_k_tokens(counter: Counter, k: int = 5) -> List[int]:
    return [tok for tok, _ in counter.most_common(k)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    token_sequences: List[List[int]],
    unigram: Counter,
    bigram_k: Dict,
    trigram_k: Dict,
    max_k: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Evaluate all three baseline models on token sequences.

    Returns:
        results[model_name][k] = {'top1_acc': float, 'top5_acc': float}
    """
    # Pre-compute unigram top-1/5 (same regardless of k)
    unigram_top1 = top_k_tokens(unigram, 1)
    unigram_top5 = top_k_tokens(unigram, 5)

    counts = {
        name: {k: {'top1_correct': 0, 'top5_correct': 0, 'total': 0}
               for k in range(1, max_k + 1)}
        for name in ('unigram', 'bigram', 'trigram')
    }

    for tokens in tqdm(token_sequences, desc="Evaluating baselines"):
        n = len(tokens)
        for k in range(1, max_k + 1):
            for i in range(n - k):
                target = tokens[i + k]
                ctx = tokens[i]
                prev_ctx = tokens[i - 1] if i >= 1 else None

                # --- Unigram ---
                c = counts['unigram'][k]
                c['total'] += 1
                if target == unigram_top1[0]:
                    c['top1_correct'] += 1
                if target in unigram_top5:
                    c['top5_correct'] += 1

                # --- Bigram ---
                c = counts['bigram'][k]
                c['total'] += 1
                bg_preds = top_k_tokens(bigram_k[k].get(ctx, Counter()), 5)
                if bg_preds and target == bg_preds[0]:
                    c['top1_correct'] += 1
                if target in bg_preds:
                    c['top5_correct'] += 1

                # --- Trigram ---
                c = counts['trigram'][k]
                c['total'] += 1
                if prev_ctx is not None:
                    tg_preds = top_k_tokens(trigram_k[k].get((prev_ctx, ctx), Counter()), 5)
                else:
                    tg_preds = []
                if tg_preds and target == tg_preds[0]:
                    c['top1_correct'] += 1
                if target in tg_preds:
                    c['top5_correct'] += 1

    results = {}
    for name in ('unigram', 'bigram', 'trigram'):
        results[name] = {}
        for k in range(1, max_k + 1):
            c = counts[name][k]
            total = c['total'] or 1
            results[name][k] = {
                'top1_acc': c['top1_correct'] / total,
                'top5_acc': c['top5_correct'] / total,
            }

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def make_results_json(
    model_name: str,
    baseline_name: str,
    metrics_by_k: Dict[int, Dict[str, float]],
    layer_indices: List[int],
    metadata: dict,
    train_dataset: str,
    val_dataset: str,
) -> dict:
    """
    Format baseline results in the experiment_results.json schema,
    replicating the same accuracy across all layer indices so they appear
    as flat horizontal lines in plot_results.sh.
    """
    results = {}
    for k, metrics in metrics_by_k.items():
        for layer in layer_indices:
            key = f"layer{layer}_k{k}"
            results[key] = {
                'layer': layer,
                'k': k,
                'val_accuracy': metrics['top1_acc'],
                'val_top5_accuracy': metrics['top5_acc'],
                # No train_accuracy or val_loss for baselines
            }

    return {
        'config': {
            'model_name': model_name,
            'baseline': baseline_name,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'max_k': max(metrics_by_k.keys()),
        },
        'metadata': metadata,
        'results': results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate n-gram baselines against look-ahead probe results"
    )
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to activations_train.pt, .tokens.jsonl, or .texts.jsonl")
    parser.add_argument("--val_dataset", type=str, required=True,
                        help="Path to activations_val.pt, .tokens.jsonl, or .texts.jsonl")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for tokenizer (only needed for .texts.jsonl or .pt inputs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write baseline result JSONs")
    parser.add_argument("--max_k", type=int, default=None,
                        help="Max lookahead distance (default: inferred from dataset)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load datasets
    # Priority: .tokens.jsonl > .texts.jsonl > .pt
    # .tokens.jsonl: raw token IDs — no tokenizer needed, exact match to model.
    # .texts.jsonl:  decoded strings — requires --model_name to re-tokenize.
    # .pt:           full activation file — also requires --model_name.
    # ------------------------------------------------------------------
    def load_tokens_and_metadata(path: str):
        """Returns (token_sequences, metadata_dict, layer_indices).

        token_sequences: List[List[int]]
        If path is .tokens.jsonl, loads IDs directly.
        If path is .texts.jsonl or .pt, re-tokenizes (requires tokenizer).
        """
        p = Path(path)
        if p.name.endswith('.tokens.jsonl'):
            token_seqs = [json.loads(line)['tokens'] for line in open(p, encoding='utf-8')]
            # Try companion .pt for metadata
            pt_path = p.with_name(p.name.replace('.tokens.jsonl', '.pt'))
            if pt_path.exists():
                meta = torch.load(pt_path, weights_only=False)['metadata']
                layers = meta.get('layers', [])
            else:
                meta, layers = {}, []
            return token_seqs, meta, layers, False  # False = no tokenizer needed
        elif p.name.endswith('.texts.jsonl'):
            texts = [json.loads(line)['text'] for line in open(p, encoding='utf-8')]
            pt_path = p.with_name(p.name.replace('.texts.jsonl', '.pt'))
            if pt_path.exists():
                meta = torch.load(pt_path, weights_only=False)['metadata']
                layers = meta.get('layers', [])
            else:
                meta, layers = {}, []
            return texts, meta, layers, True  # True = needs tokenization
        else:
            data = torch.load(path, weights_only=False)
            layers = data['metadata'].get('layers', list(data['layer_activations'].keys()))
            return data['generated_texts'], data['metadata'], layers, True

    print("Loading training dataset...")
    train_data, metadata, layer_indices, train_needs_tokenize = load_tokens_and_metadata(args.train_dataset)

    print("Loading validation dataset...")
    val_data, val_meta, val_layers, val_needs_tokenize = load_tokens_and_metadata(args.val_dataset)
    if not layer_indices:
        layer_indices = val_layers

    max_k = args.max_k or metadata.get('max_k', 1)

    needs_tokenize = train_needs_tokenize or val_needs_tokenize

    # ------------------------------------------------------------------
    # Tokenize if needed (only for .texts.jsonl or .pt inputs)
    # ------------------------------------------------------------------
    if needs_tokenize:
        if args.model_name is None:
            raise ValueError(
                "--model_name is required when loading .texts.jsonl or .pt files. "
                "Use .tokens.jsonl sidecars to skip tokenization."
            )
        from transformers import AutoTokenizer
        print(f"\nLoading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if train_needs_tokenize:
            print("Tokenizing training texts...")
            train_tokens = [
                tokenizer.encode(text, add_special_tokens=False)
                for text in tqdm(train_data, desc="Tokenizing train")
            ]
        else:
            train_tokens = train_data

        if val_needs_tokenize:
            print("Tokenizing validation texts...")
            val_tokens = [
                tokenizer.encode(text, add_special_tokens=False)
                for text in tqdm(val_data, desc="Tokenizing val")
            ]
        else:
            val_tokens = val_data
    else:
        train_tokens = train_data
        val_tokens = val_data
        print("Using raw token IDs from .tokens.jsonl (no tokenizer needed)")

    print(f"Train sequences: {len(train_tokens)}, Val sequences: {len(val_tokens)}, max_k: {max_k}")
    print(f"Layers: {len(layer_indices)} ({layer_indices[0]}–{layer_indices[-1]})" if layer_indices else "Layers: none found")

    # ------------------------------------------------------------------
    # Build n-gram tables from training set
    # ------------------------------------------------------------------
    print("\nBuilding n-gram tables...")
    unigram, bigram_k, trigram_k = build_ngram_tables(train_tokens, max_k)
    print(f"  Unigram vocabulary: {len(unigram):,} tokens")
    for k in range(1, max_k + 1):
        print(f"  Bigram contexts (k={k}): {len(bigram_k[k]):,}")
        print(f"  Trigram contexts (k={k}): {len(trigram_k[k]):,}")

    # ------------------------------------------------------------------
    # Evaluate on validation set
    # ------------------------------------------------------------------
    print("\nEvaluating on validation set...")
    results = evaluate(val_tokens, unigram, bigram_k, trigram_k, max_k)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Baseline':<12}", end="")
    for k in range(1, max_k + 1):
        print(f"  k={k} Top1    k={k} Top5 ", end="")
    print()
    print("-" * 70)
    for name in ('unigram', 'bigram', 'trigram'):
        print(f"{name:<12}", end="")
        for k in range(1, max_k + 1):
            m = results[name][k]
            print(f"  {m['top1_acc']:.4f}    {m['top5_acc']:.4f} ", end="")
        print()
    print("=" * 70)

    # ------------------------------------------------------------------
    # Write JSON files
    # ------------------------------------------------------------------
    model_name = metadata.get('model_name', args.model_name)
    for name in ('unigram', 'bigram', 'trigram'):
        data = make_results_json(
            model_name=model_name,
            baseline_name=name,
            metrics_by_k=results[name],
            layer_indices=layer_indices,
            metadata=metadata,
            train_dataset=args.train_dataset,
            val_dataset=args.val_dataset,
        )
        out_path = output_dir / f"{name}_results.json"
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {out_path}")

    print(f"\nDone. Feed any of these JSONs into plot_results.sh to overlay on probe plots.")


if __name__ == "__main__":
    main()
