"""
Regenerate the three main per-layer figures with consistent aspect ratio.

For Qwen3-32B and Gemma-3-27B the canonical aggregate.json contains 4 prompt
pairs (doom/dread, bliss/joy, dark/night, grief/pain). The 5th pair
(fright/fear) was run separately as a per-position N=100 single-prompt
experiment in {QWEN3,GEMMA3}_PER_LAYER. To produce true 5-pair means at
N=20 per pair (matching Llama-3.1-70B's run), we take the first 20
completions from each fright/fear per-position run and merge with the
4-pair aggregate.
"""

import json
import os
import pronouncing
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "..", "icml2026", "media", "results-patching")
N_FEAR = 20  # samples per layer to take from fright/fear N=100 runs

# Map pos_id -> directory token used in the {QWEN3,GEMMA3}_PER_LAYER runs
FEAR_DIR_MAP = {
    "i_minus2": "exp_fear_minus2",
    "i_minus1": "exp_fear_minus1",
    "i_0":      "exp_fear_newline",
    "i_plus1":  "exp_fear_and",
    "i_plus2":  "exp_fear_hoped",
    "i_plus3":  "exp_fear_that",
}

# (out_filename, agg_path, last_word_pos_id, last_word_label, fmt, fear_subdir_or_None)
JOBS = [
    ("patching_qwen3_32b.png",
     f"{RES}/QWEN3_AGGREGATE/qwen3_32b_aggregate_N20/aggregate.json",
     "i_minus1", "i=-1", "canonical",
     f"{RES}/QWEN3_PER_LAYER"),
    ("patching_gemma3_27b.png",
     f"{RES}/GEMMA3_AGGREGATE/gemma3_27b_aggregate_N20/aggregate.json",
     "i_minus2", "i=-2", "canonical",
     f"{RES}/GEMMA3_PER_LAYER"),
    ("patching_llama_70b.png",
     f"{RES}/llama-3.1-70b-per-layer-per-position/aggregate.json",
     "i_minus1", "i=-1", "minimal",
     None),
]


# ── Rhyme helpers (needed for re-evaluating first-N completions) ──

def _rhyme_score(w1, w2):
    p1 = pronouncing.phones_for_word(w1.lower().strip())
    p2 = pronouncing.phones_for_word(w2.lower().strip())
    if not p1 or not p2:
        return None
    rp1 = pronouncing.rhyming_part(p1[0])
    rp2 = pronouncing.rhyming_part(p2[0])
    return (rp1 == rp2) if (rp1 and rp2) else None

def _last_word(text):
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""

def _word_before_nth_newline(text, n):
    nls = [i for i, ch in enumerate(text) if ch == "\n"]
    if n <= 0 or len(nls) < n:
        return ""
    end   = nls[n - 1]
    start = nls[n - 2] + 1 if n >= 2 else 0
    return _last_word(text[start:end])

def _extract_rhyme_word(full_text, prompt):
    idx = prompt.count("\n") + 1
    w = _word_before_nth_newline(full_text, idx)
    if w:
        return w
    if full_text.startswith(prompt):
        return _last_word(full_text[len(prompt):])
    return _last_word(full_text)

def _rhyme_rate(completions, prompt, rhyme_word):
    hits = sum(1 for c in completions
               if _rhyme_score(_extract_rhyme_word(c, prompt), rhyme_word) is True)
    return hits / len(completions) if completions else 0.0


def load_fear_corrupt_rate_per_layer(fear_dir, pos_id, n=N_FEAR):
    """First-n corrupt-rhyme-rate per layer from the fright/fear single-prompt run."""
    exp_id = FEAR_DIR_MAP[pos_id]
    candidates = [d for d in os.listdir(fear_dir) if exp_id in d]
    if not candidates:
        return None
    run_dir = os.path.join(fear_dir, sorted(candidates)[0])
    with open(os.path.join(run_dir, "generations.json")) as f:
        data = json.load(f)
    clean_prompt   = data["clean_prompt"]
    corrupt_word   = data["corrupt_rhyme_word"]
    rates = []
    for layer_data in data["results"]:
        comps = layer_data["completions"][:n]
        rates.append(_rhyme_rate(comps, clean_prompt, corrupt_word))
    return rates


def load_rates(path, lw_pos, fmt, fear_dir):
    with open(path) as f:
        d = json.load(f)
    agg = d["aggregate"]
    if fmt == "canonical":
        m1_4pair = np.array([r["mean_corrupt_rhyme_rate"] for r in agg[lw_pos]])
        i0_4pair = np.array([r["mean_corrupt_rhyme_rate"] for r in agg["i_0"]])
        # If we have fright/fear data, merge to make a 5-pair mean
        if fear_dir is not None:
            fear_m1 = load_fear_corrupt_rate_per_layer(fear_dir, lw_pos)
            fear_i0 = load_fear_corrupt_rate_per_layer(fear_dir, "i_0")
            if fear_m1 is not None and fear_i0 is not None:
                fear_m1 = np.array(fear_m1)
                fear_i0 = np.array(fear_i0)
                m1 = (4 * m1_4pair + fear_m1) / 5
                i0 = (4 * i0_4pair + fear_i0) / 5
                return list(m1), list(i0)
        return list(m1_4pair), list(i0_4pair)
    else:
        m1 = [r["corrupt_rhyme_rate"] for r in agg[lw_pos]]
        i0 = [r["corrupt_rhyme_rate"] for r in agg["i_0"]]
        return m1, i0


for out_name, path, lw_pos, lw_label, fmt, fear_dir in JOBS:
    m1, i0 = load_rates(path, lw_pos, fmt, fear_dir)
    n = len(m1)
    x = np.arange(n)
    w = 0.4

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, m1, w, label=lw_label, color="#6699cc")
    ax.bar(x + w/2, i0, w, label="i=0",     color="#cc6677")
    ax.set_xlabel("Layer", fontsize=18)
    ax.set_ylabel("Corrupt rhyme rate", fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-1, n)
    step = max(1, n // 16)
    ax.set_xticks(np.arange(0, n, step))
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(OUT, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}  (peak {lw_label}: {max(m1):.3f}, peak i=0: {max(i0):.3f})")
