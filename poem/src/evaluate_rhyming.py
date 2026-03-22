#!/usr/bin/env python3
"""
Evaluate LLM rhyming accuracy on couplet completion.

For each poem prompt:
  1. Generate --n_samples completions (greedy if --temperature 0, else sampled).
  2. Extract the last alphabetic word of the generated second line.
  3. Check if it rhymes with the last word of the first line (CMU Pronouncing Dict).
  4. Report rhyme accuracy = rhymed / total_known_attempts.

Prompt format expected:
    "A rhyming couplet:\n<First Line>\n"
    or (no-trailing-newline variant)
    "A rhyming couplet:\n<First Line>"

Extraction is robust to:
  - trailing punctuation (.,;!?)
  - model-specific end-of-turn tokens (<end_of_turn>, <|im_end|>, etc.)
  - empty / newline-only continuations
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import torch
import pronouncing
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Rhyme checking
# ---------------------------------------------------------------------------

def do_rhyme(word1: str, word2: str) -> Optional[bool]:
    """
    True  → words rhyme (including identical words).
    False → words don't rhyme.
    None  → either word absent from CMU dict (unknown).
    """
    word1, word2 = word1.lower().strip(), word2.lower().strip()
    if word1 == word2:
        return True
    phones1 = pronouncing.phones_for_word(word1)
    phones2 = pronouncing.phones_for_word(word2)
    if not phones1 or not phones2:
        return None
    rp1 = {pronouncing.rhyming_part(p) for p in phones1} - {''}
    rp2 = {pronouncing.rhyming_part(p) for p in phones2} - {''}
    return bool(rp1 & rp2)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r'<[^>]+>')


def _last_alpha_word(text: str) -> Optional[str]:
    """Return the last run of letters in text, lower-cased."""
    words = re.findall(r'[a-zA-Z]+', text)
    return words[-1].lower() if words else None


def extract_first_line_word(prompt: str) -> Optional[str]:
    """
    Last alphabetic word of the first couplet line.
    Works for prompts with or without a trailing newline.
    """
    lines = [l for l in prompt.split('\n') if l.strip()]
    # lines[0] = "A rhyming couplet:", lines[1] = first couplet line
    if len(lines) < 2:
        return None
    return _last_alpha_word(lines[1])


def extract_second_line_word(continuation: str) -> Optional[str]:
    """
    Extract the rhyme word from a model continuation.

    Stops at the first of:
      - a newline character
      - a model end-of-turn token (<end_of_turn>, <|im_end|>, etc.)

    Then returns the last alphabetic word in that segment.
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_completions(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    device,
) -> List[str]:
    """
    Return a list of n_samples decoded continuations (prompt tokens stripped).
    Uses greedy decoding when temperature == 0, sampling otherwise.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    greedy = temperature == 0.0

    if greedy:
        # Single greedy decode
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        cont_ids = out[0, prompt_len:]
        return [tokenizer.decode(cont_ids, skip_special_tokens=False)]
    else:
        # Batch n_samples in one forward pass
        input_ids_rep = input_ids.expand(n_samples, -1)
        with torch.no_grad():
            out = model.generate(
                input_ids_rep,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=pad_token_id,
            )
        return [
            tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
            for i in range(n_samples)
        ]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    prompts: List[str],
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    device,
) -> dict:
    """
    Evaluate rhyming accuracy over prompts.

    Returns:
        n_total       total prompts processed
        n_attempted   prompts where both words were extractable (per-sample)
        n_rhymed      of n_attempted, how many rhymed (CMU known + rhyme=True)
        n_unknown     of n_attempted, how many had a CMU-unknown word
        rhyme_rate    n_rhymed / n_total  (denominator = total poems)
        results       list of per-poem dicts
    """
    greedy = temperature == 0.0
    effective_n = 1 if greedy else n_samples

    poem_results = []
    n_rhymed = 0
    n_unknown = 0
    n_attempted = 0  # counts per sample

    model.eval()

    for prompt in tqdm(prompts, desc="Evaluating"):
        first_word = extract_first_line_word(prompt)
        if first_word is None:
            poem_results.append({'status': 'skip_no_first_word', 'prompt': prompt})
            continue

        continuations = generate_completions(
            model, tokenizer, prompt, effective_n, temperature, max_new_tokens, device
        )

        sample_records = []
        for cont in continuations:
            second_word = extract_second_line_word(cont)
            if second_word is None:
                sample_records.append({
                    'status': 'skip_no_second_word',
                    'continuation': cont,
                })
                continue

            n_attempted += 1
            result = do_rhyme(first_word, second_word)
            if result is None:
                n_unknown += 1
                status = 'unknown'
            elif result:
                n_rhymed += 1
                status = 'rhyme'
            else:
                status = 'no_rhyme'

            sample_records.append({
                'status': status,
                'second_word': second_word,
                'continuation': cont,
            })

        poem_results.append({
            'first_word': first_word,
            'prompt': prompt,
            'samples': sample_records,
        })

    n_total = len(prompts)
    return {
        'n_total':     n_total,
        'n_attempted': n_attempted,
        'n_rhymed':    n_rhymed,
        'n_unknown':   n_unknown,
        'rhyme_rate':  n_rhymed / n_total if n_total else 0.0,
        'results':     poem_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM rhyming accuracy on couplet completion"
    )
    parser.add_argument("--model_name",     type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--poems_path",     type=str, required=True,
                        help="Path to JSONL file with poem prompts (field: text)")
    parser.add_argument("--output_dir",     type=str, default=None,
                        help="Directory to write results.json (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_poems",      type=int, default=None,
                        help="Cap number of poems (default: all)")
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="Sampling temperature; 0 = greedy (default: 0)")
    parser.add_argument("--n_samples",      type=int, default=1,
                        help="Completions per poem when temperature > 0 (default: 1)")
    parser.add_argument("--device",         type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quantization",   type=str, default=None,
                        choices=["4bit", "8bit"],
                        help="Quantize model: '8bit' halves bfloat16 memory, "
                             "'4bit' quarters it (requires bitsandbytes)")
    args = parser.parse_args()

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

    # Build quantization config
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
    input_device = next(model.parameters()).device
    print(f"  Loaded on device: {input_device}\n")

    # Determine effective sampling settings
    greedy = args.temperature == 0.0
    effective_n = 1 if greedy else args.n_samples
    print(f"Mode:        {'greedy' if greedy else f'sampling (temp={args.temperature}, n={effective_n})'}")
    print(f"max_new_tokens: {args.max_new_tokens}\n")

    out = evaluate(
        model, tokenizer, prompts,
        n_samples=effective_n,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=input_device,
    )

    # Summary table
    print()
    print("=" * 52)
    print(f"  Total poems:    {out['n_total']}")
    print(f"  Attempted:      {out['n_attempted']}  (word extracted from both lines)")
    print(f"  Rhymed:         {out['n_rhymed']}")
    print(f"  Unknown (CMU):  {out['n_unknown']}")
    print(f"  Rhyme rate:     {out['rhyme_rate']*100:.1f}%  (rhymed / total poems)")
    print("=" * 52)

    if args.output_dir:
        out_path = Path(args.output_dir) / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_name':     args.model_name,
            'temperature':    args.temperature,
            'n_samples':      effective_n,
            'max_new_tokens': args.max_new_tokens,
            **{k: out[k] for k in ('n_total', 'n_attempted', 'n_rhymed', 'n_unknown', 'rhyme_rate')},
            'results':        out['results'],
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
