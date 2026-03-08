"""
Rhyme evaluation using the CMU Pronouncing Dictionary.

A rhyme key is defined as: last stressed vowel phoneme + all following phonemes.
Two words rhyme iff their rhyme keys are equal.
"""
import re
from typing import Optional, List

try:
    import pronouncing
    _HAS_PRONOUNCING = True
except ImportError:
    _HAS_PRONOUNCING = False
    print("Warning: 'pronouncing' not installed — rhyme evaluation will always return False.")


def get_rhyme_key(word: str) -> Optional[str]:
    """Return the rhyme key for a word, or None if not in CMU dict."""
    if not _HAS_PRONOUNCING:
        return None
    phones_list = pronouncing.phones_for_word(word.lower())
    if not phones_list:
        return None
    phones = phones_list[0].split()
    for i in range(len(phones) - 1, -1, -1):
        if phones[i][-1] in ("1", "2"):
            return " ".join(phones[i:])
    return " ".join(phones)  # fallback: all phones (unstressed word)


def get_last_word(text: str) -> Optional[str]:
    """Extract the last alphabetic word from a string."""
    words = re.findall(r"[a-zA-Z']+", text)
    return words[-1].lower() if words else None


def rhyme_key_of_text(text: str) -> Optional[str]:
    word = get_last_word(text)
    return get_rhyme_key(word) if word else None


def scheme_rhyme_key(texts: List[str]) -> Optional[str]:
    """
    Infer the consensus rhyme key for a group of couplet-first-lines.
    Uses the most common rhyme key among their last words.
    """
    from collections import Counter
    keys = []
    for t in texts:
        # Strip trailing whitespace/newlines to get last meaningful word
        key = rhyme_key_of_text(t.rstrip())
        if key:
            keys.append(key)
    if not keys:
        return None
    return Counter(keys).most_common(1)[0][0]


def evaluate_rhyme(generated_text: str, target_rhyme_key: str) -> bool:
    """
    Return True if the last word of generated_text matches target_rhyme_key.
    """
    key = rhyme_key_of_text(generated_text)
    if key is None:
        return False
    return key == target_rhyme_key
