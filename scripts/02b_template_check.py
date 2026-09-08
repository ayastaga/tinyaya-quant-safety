"""Phase 0 BLOCKING GATE: verify the GGUF chat template matches transformers.

Runs the same prompts through BF16 (transformers) and Q8_0 (llama.cpp). Q8_0 is
near-lossless, so with greedy decoding the responses should be near-identical.
Large divergence => template/conversion bug => EVERY delta in the project is
garbage. Fix before proceeding.

Usage: python 02b_template_check.py --model global
Exit code 1 on failure.
"""
import argparse
import difflib
import sys

from common import load_config

PROBES = [
    "What is the capital of France? Answer in one word.",
    "2+2=? Reply with just the number.",
    "List three colors, comma-separated.",
    "Translate 'good morning' into Spanish. Only the translation.",
    "Habari yako? Jibu kwa sentensi moja.",  # Swahili probe
]

PASS_RATIO = 0.85  # mean similarity threshold; near-identical expected at Q8_0


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="global")
    args = ap.parse_args()

    # local imports so this file parses without heavy deps
    from importlib import import_module
    gen = import_module("03_generate")  # same directory; digit-leading name is fine via importlib
    hf = gen.HFBackend(cfg, args.model)
    gg = gen.GGUFBackend(cfg, args.model, "q8_0")

    sims = []
    for p in PROBES:
        a = hf.generate(p).strip()
        b = gg.generate(p).strip()
        sim = difflib.SequenceMatcher(None, a, b).ratio()
        sims.append(sim)
        print(f"\nPROBE: {p}\n  BF16 : {a[:200]}\n  Q8_0 : {b[:200]}\n  sim={sim:.2f}")

    mean_sim = sum(sims) / len(sims)
    print(f"\nMean similarity: {mean_sim:.2f} (threshold {PASS_RATIO})")
    if mean_sim < PASS_RATIO:
        print("FAIL: template/conversion mismatch likely. Do NOT proceed; "
              "inspect chat template in the GGUF (llama.cpp logs it at load) "
              "vs tokenizer_config.json, and convert_hf_to_gguf warnings.")
        sys.exit(1)
    print("PASS: record this in README (Phase 0 step 5).")


if __name__ == "__main__":
    main()
