#!/usr/bin/env python3
"""
Ablation: evaluate how often a model produces a rhyming second line.

Dataset: poems-original-truncated-shuffled.jsonl
Format:  {"id": ..., "text": "A rhyming couplet:\n<First Line>,\n"}

For each poem:
  1. Feed the prompt to the model (max_new_tokens=16).
  2. Extract the last word of the generated second line.
  3. Check if it rhymes with the last word of the first line.
  4. Report (rhymed) / (total attempted).

Two extraction modes (--mode):
  with_newline
      Normal generation. The second line ends at the first \\n in the
      continuation. Strip trailing punctuation, take the last alphabetic word.
      Handles ".\n" fusion: the string-level approach automatically ignores it.

  without_newline
      Suppress \\n from generation via bad_words_ids. Find the first
      punctuation mark (. ! ? ; ,) in the continuation and take everything
      before it as the second line.

Rhyme check: CMU Pronouncing Dictionary via the `pronouncing` library.
  - Words absent from the CMU dict are counted as "unknown" (excluded from
    the rhyme rate denominator if --strict-known is set, otherwise counted
    as non-rhyming).
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import torch
import pronouncing
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


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


_PUNCT_RE = re.compile(r'[.!?;,]')


def extract_second_line_word_with_newline(continuation: str) -> Optional[str]:
    """
    Extract the rhyme word from a continuation that terminates at \\n.
    Takes everything up to the first \\n (or the full string if absent),
    strips trailing punctuation, and returns the last alphabetic word.
    Handles the ".<newline>" BPE-fusion case at string level automatically.
    """
    nl_pos = continuation.find('\n')
    segment = continuation[:nl_pos] if nl_pos >= 0 else continuation
    return _last_alpha_word(segment)


def extract_second_line_word_without_newline(continuation: str) -> Optional[str]:
    """
    Extract the rhyme word when \\n is suppressed during generation.
    Finds the first sentence-ending punctuation mark and takes everything
    before it as the second line.
    """
    m = _PUNCT_RE.search(continuation)
    segment = continuation[:m.start()] if m else continuation
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
) -> dict:
    """
    Evaluate rhyming performance over `prompts`.

    Returns a dict:
        n_total      : number of prompts processed
        n_attempted  : prompts where both words were successfully extracted
        n_rhymed     : of n_attempted, how many rhymed (CMU known + rhyme=True)
        n_unknown    : of n_attempted, how many had a CMU-unknown word
        results      : list of per-poem dicts (status, first_word, second_word,
                       continuation)
    """
    assert mode in ('with_newline', 'without_newline'), f"Unknown mode: {mode}"

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

    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Evaluating ({mode})"):
            first_word = extract_first_line_word(prompt)
            if first_word is None:
                results.append({'status': 'skip_no_first_word'})
                continue

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
            if bad_words_ids:
                gen_kwargs['bad_words_ids'] = bad_words_ids

            generated_ids = model.generate(input_ids, **gen_kwargs)
            continuation_ids = generated_ids[0, input_ids.shape[1]:]
            continuation = tokenizer.decode(continuation_ids, skip_special_tokens=False)

            if mode == 'with_newline':
                second_word = extract_second_line_word_with_newline(continuation)
            else:
                second_word = extract_second_line_word_without_newline(continuation)

            if second_word is None:
                results.append({
                    'status': 'skip_no_second_word',
                    'first_word': first_word,
                    'continuation': continuation,
                })
                continue

            n_attempted += 1
            rhyme_result = do_rhyme(first_word, second_word)

            if rhyme_result is None:
                n_unknown += 1
                status = 'unknown'
            elif rhyme_result:
                n_rhymed += 1
                status = 'rhyme'
            else:
                status = 'no_rhyme'

            results.append({
                'status': status,
                'first_word': first_word,
                'second_word': second_word,
                'continuation': continuation,
            })

    return {
        'n_total': len(prompts),
        'n_attempted': n_attempted,
        'n_rhymed': n_rhymed,
        'n_unknown': n_unknown,
        'results': results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save_result(out: dict, mode: str, model_name: str, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'mode':        mode,
        'model_name':  model_name,
        'n_total':     out['n_total'],
        'n_attempted': out['n_attempted'],
        'n_rhymed':    out['n_rhymed'],
        'n_unknown':   out['n_unknown'],
        'rhyme_rate':  out['n_rhymed'] / out['n_total'] if out['n_total'] else 0,
        'results':     out['results'],
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
    # device_map="auto" places shards directly on GPU(s) during loading,
    # avoiding a separate .to(device) call that can hit CUDA error 802.
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Determine the device to place input tensors on (first shard's device)
    input_device = next(model.parameters()).device
    print(f"✓ Loaded (input device: {input_device})\n")

    # Run each mode and collect results
    all_results = {}
    for mode in modes:
        out = evaluate(
            model, tokenizer, prompts,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            device=input_device,
        )
        all_results[mode] = out

        if args.output_dir:
            output_path = str(Path(args.output_dir) / f"results_{mode}.json")
            _save_result(out, mode, args.model_name, output_path)

    # Combined accuracy table
    print()
    print("=" * 52)
    print(f"{'Mode':<20}  {'Rhymed':>7}  {'Attempted':>9}  {'Accuracy':>8}")
    print("-" * 52)
    for mode, out in all_results.items():
        r = out['n_rhymed']
        n = out['n_attempted']
        total = out['n_total']
        acc = f"{r/total*100:.1f}%" if total else "—"
        print(f"{mode:<20}  {r:>7}  {n:>9}  {acc:>8}")
    print("=" * 52)
    print("(Accuracy = rhymed / total poems)")


if __name__ == "__main__":
    main()
