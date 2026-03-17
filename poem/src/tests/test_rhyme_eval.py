#!/usr/bin/env python3
"""
Playground for testing rhyme word extraction and rhyme checking.

Run directly:
    python poem/src/tests/test_rhyme_eval.py

Add your own cases to EXTRACTION_CASES or RHYME_CASES at the bottom.
"""

import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ablation.evaluate_rhyming import (
    extract_first_line_word,
    extract_second_line_word_with_newline,
    extract_second_line_word_without_newline,
    do_rhyme,
)


# ---------------------------------------------------------------------------
# Extraction test cases
# format: (description, mode, input, expected_output)
#   mode: "first_line" | "with_newline" | "without_newline"
# ---------------------------------------------------------------------------
EXTRACTION_CASES = [
    # --- Normal \n termination ---
    ("normal \\n termination",
     "with_newline",
     "hoped to stay the night.\n",
     "night"),

    ("normal \\n, no punctuation",
     "with_newline",
     "she danced beneath the moon\n",
     "moon"),

    # --- Gemma: <end_of_turn> instead of \n ---
    ("Gemma <end_of_turn>",
     "with_newline",
     "hoped to stay the night.<end_of_turn>",
     "night"),

    ("Gemma <end_of_turn> no punctuation",
     "with_newline",
     "she danced beneath the moon<end_of_turn>",
     "moon"),

    # --- Qwen: <|im_end|> ---
    ("Qwen <|im_end|>",
     "with_newline",
     "the stars shone bright all night.<|im_end|>",
     "night"),

    # --- Trailing garbage after special token shouldn't affect result ---
    ("special token followed by more text",
     "with_newline",
     "hoped to stay the night.<end_of_turn>\nsome extra stuff",
     "night"),

    # --- without_newline mode ---
    ("without_newline: stops at punctuation",
     "without_newline",
     "hoped to stay the night.<end_of_turn>",
     "night"),

    ("without_newline: no punctuation, with special token",
     "without_newline",
     "she danced beneath the moon<end_of_turn>",
     "moon"),

    # --- first_line extraction ---
    ("first line extraction",
     "first_line",
     "A rhyming couplet:\nThe children played at night\n",
     "night"),

    ("first line extraction — trailing punctuation",
     "first_line",
     "A rhyming couplet:\nAll the stars burned bright,\n",
     "bright"),

    # --- Edge cases ---
    ("empty continuation",
     "with_newline",
     "",
     None),

    ("only special token",
     "with_newline",
     "<end_of_turn>",
     None),

    ("multi-word second line, special token",
     "with_newline",
     "roses are red and the sky is blue<end_of_turn>",
     "blue"),
]


# ---------------------------------------------------------------------------
# Rhyme check cases
# format: (word1, word2, expected)  — expected: True | False | None (unknown)
# ---------------------------------------------------------------------------
RHYME_CASES = [
    ("night",  "light",  True),
    ("night",  "moon",   False),
    ("moon",   "june",   True),
    ("sky",    "fly",    True),
    ("blue",   "true",   True),
    ("blue",   "night",  False),
    ("day",    "say",    True),
    ("love",   "move",   False),  # slant rhyme — CMU dict does not consider these rhyming
    ("orange", "purple", False),
    # unknown words
    ("zxqwy",  "night",  None),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_extraction():
    print("=" * 60)
    print("EXTRACTION TESTS")
    print("=" * 60)
    passed = failed = 0
    for desc, mode, inp, expected in EXTRACTION_CASES:
        if mode == "first_line":
            got = extract_first_line_word(inp)
        elif mode == "with_newline":
            got = extract_second_line_word_with_newline(inp)
        elif mode == "without_newline":
            got = extract_second_line_word_without_newline(inp)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ok = got == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         input:    {inp!r}")
            print(f"         expected: {expected!r}")
            print(f"         got:      {got!r}")

    print(f"\n  {passed}/{passed+failed} passed\n")
    return failed


def _run_rhyme():
    print("=" * 60)
    print("RHYME CHECKS")
    print("=" * 60)
    passed = failed = 0
    for word1, word2, expected in RHYME_CASES:
        got = do_rhyme(word1, word2)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        label = {True: "rhyme", False: "no rhyme", None: "unknown"}
        exp_str = label[expected]
        got_str = label[got]
        print(f"  [{status}] {word1!r:10} + {word2!r:10}  expected={exp_str:10}  got={got_str}")

    print(f"\n  {passed}/{passed+failed} passed\n")
    return failed


if __name__ == "__main__":
    n_failed = 0
    n_failed += _run_extraction()
    n_failed += _run_rhyme()

    print("=" * 60)
    if n_failed == 0:
        print("All tests passed.")
    else:
        print(f"{n_failed} test(s) FAILED.")
    sys.exit(1 if n_failed else 0)
