#!/usr/bin/env python3
"""
Ablation: evaluate how often a model produces a rhyming second line.

Dataset: poems-original-truncated-shuffled.jsonl
Format:  {"id": ..., "text": "A rhyming couplet:\n<First Line>,\n"}

For each poem:
  1. Feed the prompt to the model (max_new_tokens=16).
  2. Extract the last word of the generated second line.
  3. Check if it rhymes with the last word of the first line.
  4. Report (rhymed) / (total attempted) with 95% Wilson confidence interval.

Two extraction modes (--mode):
  with_newline
      Normal generation. The second line ends at the first \\n in the
      continuation. Strip trailing punctuation, take the last alphabetic word.
      Handles ".\n" fusion: the string-level approach automatically ignores it.

  without_newline
      Suppress \\n from generation via bad_words_ids. Find the first
      punctuation mark (. ! ? ; ,) in the continuation and take everything
      before it as the second line.

Temperature sampling:
  --temperature > 0 enables sampling; --n_samples draws that many completions
  per poem. Rhyme rate is computed over all (poem × sample) pairs.
  Per-poem mean and std are reported alongside the overall Wilson CI.

Rhyme check: CMU Pronouncing Dictionary via the `pronouncing` library.
  - Words absent from the CMU dict are counted as "unknown" (excluded from
    the rhyme rate denominator if --strict-known is set, otherwise counted
    as non-rhyming).
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Optional

import torch
import pronouncing
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mean_std(values: List[float]):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


# ---------------------------------------------------------------------------
# Rhyme checking
# ---------------------------------------------------------------------------

def do_rhyme(word1: str, word2: str) -> Optional[bool]:
    """
    Return True if word1 and word2 rhyme, False if they don't,
    None if either word is absent from the CMU Pronouncing Dictionary.
    """
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()
    if word1 == word2:
        return True  # identical words count as rhyming
    phones1 = pronouncing.phones_for_word(word1)
    phones2 = pronouncing.phones_for_word(word2)
    if not phones1 or not phones2:
        return None
    rp1 = {pronouncing.rhyming_part(p) for p in phones1} - {''}
    rp2 = {pronouncing.rhyming_part(p) for p in phones2} - {''}
    return bool(rp1 & rp2)


# ---------------------------------------------------------------------------
# Word extraction helpers
# ---------------------------------------------------------------------------

def _last_alpha_word(text: str) -> Optional[str]:
    """Return the last run of letters in `text`, lower-cased."""
    words = re.findall(r'[a-zA-Z]+', text)
    return words[-1].lower() if words else None


def extract_first_line_word(prompt_text: str) -> Optional[str]:
    """
    Extract the last word of the first couplet line from the prompt.
    Prompt format: "A rhyming couplet:\\n<First Line>\\n"
    """
    parts = prompt_text.rstrip('\n').split('\n')
    if len(parts) < 2:
        return None
    return _last_alpha_word(parts[-1])


_PUNCT_RE = re.compile(r'[.!?]')
# Matches special tokens like <end_of_turn> (Gemma) or <|im_end|> (Qwen)
_SPECIAL_TOKEN_RE = re.compile(r'<[^>]+>')


def extract_second_line_word_with_newline(continuation: str) -> Optional[str]:
    """
    Extract the rhyme word from a continuation that terminates at \\n or a
    model-specific end-of-turn token (e.g. Gemma's <end_of_turn>, Qwen's
    <|im_end|>).  Takes everything up to the first terminator (or the full
    string if absent), then returns the last alphabetic word.
    """
    end = len(continuation)
    nl_pos = continuation.find('\n')
    if nl_pos >= 0:
        end = min(end, nl_pos)
    st_m = _SPECIAL_TOKEN_RE.search(continuation)
    if st_m:
        end = min(end, st_m.start())
    segment = continuation[:end]
    return _last_alpha_word(segment)


def extract_second_line_word_without_newline(continuation: str) -> Optional[str]:
    """
    Extract the rhyme word when \\n is suppressed during generation.
    Finds the first sentence-ending punctuation mark or special token
    (e.g. <end_of_turn>, <|im_end|>) and takes everything before it.
    """
    end = len(continuation)
    m = _PUNCT_RE.search(continuation)
    if m:
        end = min(end, m.start())
    st_m = _SPECIAL_TOKEN_RE.search(continuation)
    if st_m:
        end = min(end, st_m.start())
    segment = continuation[:end]
    return _last_alpha_word(segment)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _newline_bad_word_ids(tokenizer) -> List[List[int]]:
    """
    Return bad_words_ids for every token whose decoded form contains \\n.
    """
    bad = []
    for tid in range(tokenizer.vocab_size):
        if '\n' in tokenizer.decode([tid], skip_special_tokens=False):
            bad.append([tid])
    return bad


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    prompts: List[str],
    mode: str,
    max_new_tokens: int = 16,
    device: str = "cuda",
    temperature: float = 0.0,
    n_samples: int = 1,
) -> dict:
    """
    Evaluate rhyming performance over `prompts`.

    When temperature > 0 and n_samples > 1, draws n_samples completions per
    prompt and aggregates rhyme counts across all (poem × sample) pairs.

    Returns a dict with overall counts, Wilson CI, and per-poem stats.
    """
    assert mode in ('with_newline', 'without_newline'), f"Unknown mode: {mode}"

    do_sampling = temperature > 0.0 and n_samples > 1
    effective_n_samples = n_samples if do_sampling else 1

    bad_words_ids = None
    if mode == 'without_newline':
        print("Computing bad_words_ids for newline suppression…")
        bad_words_ids = _newline_bad_word_ids(tokenizer)
        print(f"  Suppressing {len(bad_words_ids)} token(s) that contain '\\n'")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.eval()

    results = []
    n_rhymed = 0
    n_unknown = 0
    n_attempted = 0
    per_poem_rates = []   # rhyme rate per poem (for mean/std across poems)

    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Evaluating ({mode})"):
            first_word = extract_first_line_word(prompt)
            if first_word is None:
                results.append({'status': 'skip_no_first_word'})
                continue

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
            )
            if do_sampling:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=effective_n_samples,
                )
            else:
                gen_kwargs['do_sample'] = False

            if bad_words_ids:
                gen_kwargs['bad_words_ids'] = bad_words_ids

            # generated_ids: [n_samples, seq_len]
            generated_ids = model.generate(
                input_ids.expand(effective_n_samples, -1) if do_sampling else input_ids,
                **gen_kwargs,
            )

            prompt_len = input_ids.shape[1]
            poem_rhymed = 0
            poem_attempted = 0
            sample_results = []

            for i in range(effective_n_samples):
                continuation_ids = generated_ids[i, prompt_len:]
                continuation = tokenizer.decode(continuation_ids, skip_special_tokens=False)

                if mode == 'with_newline':
                    second_word = extract_second_line_word_with_newline(continuation)
                else:
                    second_word = extract_second_line_word_without_newline(continuation)

                if second_word is None:
                    sample_results.append({
                        'status': 'skip_no_second_word',
                        'continuation': continuation,
                    })
                    continue

                n_attempted += 1
                poem_attempted += 1
                rhyme_result = do_rhyme(first_word, second_word)

                if rhyme_result is None:
                    n_unknown += 1
                    status = 'unknown'
                elif rhyme_result:
                    n_rhymed += 1
                    poem_rhymed += 1
                    status = 'rhyme'
                else:
                    status = 'no_rhyme'

                sample_results.append({
                    'status': status,
                    'second_word': second_word,
                    'continuation': continuation,
                })

            poem_rate = poem_rhymed / poem_attempted if poem_attempted > 0 else None
            if poem_rate is not None:
                per_poem_rates.append(poem_rate)

            results.append({
                'first_word': first_word,
                'poem_rhymed': poem_rhymed,
                'poem_attempted': poem_attempted,
                'poem_rhyme_rate': poem_rate,
                'samples': sample_results,
            })

    ci_low, ci_high = wilson_ci(n_rhymed, n_attempted)
    poem_mean, poem_std = mean_std(per_poem_rates)

    return {
        'n_total':        len(prompts),
        'n_attempted':    n_attempted,
        'n_rhymed':       n_rhymed,
        'n_unknown':      n_unknown,
        'n_samples':      effective_n_samples,
        'temperature':    temperature,
        'rhyme_rate':     n_rhymed / n_attempted if n_attempted else 0.0,
        'ci_low':         ci_low,
        'ci_high':        ci_high,
        'per_poem_mean':  poem_mean,
        'per_poem_std':   poem_std,
        'results':        results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save_result(out: dict, mode: str, model_name: str, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'mode':           mode,
        'model_name':     model_name,
        'n_total':        out['n_total'],
        'n_attempted':    out['n_attempted'],
        'n_rhymed':       out['n_rhymed'],
        'n_unknown':      out['n_unknown'],
        'n_samples':      out['n_samples'],
        'temperature':    out['temperature'],
        'rhyme_rate':     out['rhyme_rate'],
        'ci_low':         out['ci_low'],
        'ci_high':        out['ci_high'],
        'per_poem_mean':  out['per_poem_mean'],
        'per_poem_std':   out['per_poem_std'],
        'results':        out['results'],
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM rhyming performance on couplet completion"
    )
    parser.add_argument("--model_name",     type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--poems_path",     type=str, required=True,
                        help="Path to JSONL file with poem prompts")
    parser.add_argument("--output_dir",     type=str, default=None,
                        help="Directory to write per-mode JSON results (optional)")
    parser.add_argument("--mode",           type=str, default="both",
                        choices=["with_newline", "without_newline", "both"],
                        help="Which mode(s) to run (default: both)")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_poems",      type=int, default=None,
                        help="Cap number of poems (default: all)")
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="Sampling temperature; 0 = greedy (default: 0)")
    parser.add_argument("--n_samples",      type=int, default=1,
                        help="Completions per poem when temperature > 0 (default: 1)")
    args = parser.parse_args()

    modes = ["with_newline", "without_newline"] if args.mode == "both" else [args.mode]

    # Load prompts
    prompts = []
    with open(args.poems_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)['text'])
    if args.max_poems:
        prompts = prompts[:args.max_poems]
    print(f"Loaded {len(prompts)} prompts from {args.poems_path}")

    # Load model + tokenizer once
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    input_device = next(model.parameters()).device
    print(f"✓ Loaded (input device: {input_device})\n")

    # Run each mode
    all_results = {}
    for mode in modes:
        out = evaluate(
            model, tokenizer, prompts,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            device=input_device,
            temperature=args.temperature,
            n_samples=args.n_samples,
        )
        all_results[mode] = out

        if args.output_dir:
            output_path = str(Path(args.output_dir) / f"results_{mode}.json")
            _save_result(out, mode, args.model_name, output_path)

    # Summary table
    col = 22
    print()
    print("=" * 72)
    print(f"{'Mode':<{col}}  {'Rhymed':>7}  {'Attempted':>9}  {'Rate':>6}  {'95% CI':>18}  {'Per-poem std':>12}")
    print("-" * 72)
    for mode, out in all_results.items():
        r   = out['n_rhymed']
        n   = out['n_attempted']
        rate = out['rhyme_rate']
        lo, hi = out['ci_low'], out['ci_high']
        std = out['per_poem_std']
        ci_str = f"[{lo:.3f}, {hi:.3f}]"
        std_str = f"±{std:.3f}" if out['n_samples'] > 1 else "—"
        print(f"{mode:<{col}}  {r:>7}  {n:>9}  {rate:>6.3f}  {ci_str:>18}  {std_str:>12}")
    print("=" * 72)
    sampling_note = (f"n_samples={args.n_samples}, temperature={args.temperature}"
                     if args.temperature > 0 else "greedy decoding")
    print(f"({sampling_note}; CI = Wilson 95%; per-poem std across poems)")


if __name__ == "__main__":
    main()
