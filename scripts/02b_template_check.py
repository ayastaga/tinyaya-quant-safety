"""Phase 0 BLOCKING GATE: BF16 (transformers) vs Q8_0 (llama.cpp) parity.

Q8_0 is near-lossless, so with greedy decoding the two backends should agree.
Divergence means a template / tokenization / stop-token bug, and then EVERY
delta in the project is garbage. Fix before proceeding.

Per probe this checks four things, all hard failures:
  1. prompt token ids identical in transformers and llama.cpp
     (same chat rendering AND same tokenization of it);
  2. both backends stopped on a recognized stop token, not on max_tokens;
  3. neither response contains obvious repetition (the max_tokens loop
     signature, and what a missed EOG token produces);
  4. the two responses are similar enough (mean similarity >= PASS_RATIO).

Nothing here writes to the HF checkpoint or the GGUF; both are read-only.

Usage: python 02b_template_check.py --model global [--precision q8_0]
Exit code 1 if any probe fails.
"""
import argparse
import difflib
import sys

from common import llama_cpp_pin, load_config

PROBES = [
    "What is the capital of France? Answer in one word.",
    "2+2=? Reply with just the number.",
    "List three colors, comma-separated.",
    "Translate 'good morning' into Spanish. Only the translation.",
    "Habari yako? Jibu kwa sentensi moja.",           # Swahili probe
    "用一句话解释什么是光合作用。",                      # Chinese probe (no spaces)
    "Write two sentences about the sea.",             # multi-sentence: repetition bait
]

PASS_RATIO = 0.70   # mean similarity threshold; near-identical expected at Q8_0
PROBE_BUDGET = 256  # small budget: a healthy turn stops long before this


def looks_repetitive(text):
    """Detect the degenerate loop a missed stop token produces.

    Returns (is_repetitive, evidence). Word n-grams catch space-separated
    scripts; the character pass catches zh/ja/th, where .split() is useless.
    """
    t = " ".join(text.split())
    if len(t) < 80:
        return False, ""
    words = t.split()
    if len(words) >= 24:
        counts = {}
        for i in range(len(words) - 7):
            g = " ".join(words[i:i + 8])
            counts[g] = counts.get(g, 0) + 1
        g, c = max(counts.items(), key=lambda kv: kv[1])
        if c >= 3:
            return True, f"8-gram x{c}: {g[:60]!r}"
    counts = {}
    for i in range(len(t) - 19):
        g = t[i:i + 20]
        counts[g] = counts.get(g, 0) + 1
    if counts:
        g, c = max(counts.items(), key=lambda kv: kv[1])
        if c >= 3:
            return True, f"20-char x{c}: {g[:40]!r}"
    return False, ""


def check_side(name, res, stop_strings):
    """Per-backend hard checks on one generation. Returns a list of failures."""
    fails = []
    if res["stop_reason"] != "eog":
        fails.append(f"{name}: stop_reason={res['stop_reason']!r} "
                     f"(no stop token in {res['n_generated_tokens']} tokens)")
    elif res["stop_token"] not in stop_strings:
        fails.append(f"{name}: unrecognized stop token "
                     f"{res['stop_token']!r} (id={res['stop_token_id']})")
    rep, evidence = looks_repetitive(res["response"])
    if rep:
        fails.append(f"{name}: repetition detected -- {evidence}")
    return fails


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="global")
    ap.add_argument("--precision", default="q8_0")
    args = ap.parse_args()

    # local imports so this file parses without heavy deps
    from importlib import import_module
    gen = import_module("03_generate")  # same directory; digit-leading name is fine via importlib

    print(f"model={args.model} precision={args.precision} "
          f"probes={len(PROBES)} llama.cpp pin: {llama_cpp_pin()}")
    hf = gen.HFBackend(cfg, args.model)
    gg = gen.GGUFBackend(cfg, args.model, args.precision)
    stop_strings = set(gen.GGUF_STOP_TOKEN_STRINGS)

    sims, failures = [], []
    for p in PROBES:
        probe_fails = []

        hf_rendered, hf_ids = hf.prompt_token_ids(p)
        gg_rendered, gg_ids = gg.prompt_token_ids(p)
        ids_match = hf_ids == gg_ids
        if hf_rendered != gg_rendered:
            probe_fails.append("rendered prompt strings differ between backends")
        if not ids_match:
            diff = next((i for i, (a, b) in enumerate(zip(hf_ids, gg_ids)) if a != b),
                        min(len(hf_ids), len(gg_ids)))
            probe_fails.append(
                f"prompt token ids differ (hf={len(hf_ids)} tok, llama={len(gg_ids)} tok, "
                f"first difference at index {diff}: "
                f"{hf_ids[diff:diff + 4]} vs {gg_ids[diff:diff + 4]})")

        a = hf.generate(p, max_new_tokens=PROBE_BUDGET)
        b = gg.generate(p, max_new_tokens=PROBE_BUDGET)
        probe_fails += check_side("BF16", a, stop_strings)
        probe_fails += check_side(args.precision.upper(), b, stop_strings)
        if b["prompt_tokenization_match"] is False:
            probe_fails.append("backend reported prompt_tokenization_match=False")

        sim = difflib.SequenceMatcher(
            None, a["response"].strip(), b["response"].strip()).ratio()
        sims.append(sim)

        print(f"\nPROBE: {p}"
              f"\n  prompt tokens: {len(hf_ids)} "
              f"({'identical' if ids_match else 'MISMATCH'})"
              f"\n  BF16 : {a['response'].strip()[:200]}"
              f"\n         stop={a['stop_token']}({a['stop_token_id']}) "
              f"reason={a['stop_reason']} n={a['n_generated_tokens']}"
              f"\n  {args.precision.upper():<5}: {b['response'].strip()[:200]}"
              f"\n         stop={b['stop_token']}({b['stop_token_id']}) "
              f"reason={b['stop_reason']} n={b['n_generated_tokens']}"
              f"\n  sim={sim:.2f}")
        for f in probe_fails:
            print(f"  FAIL {f}")
        failures += [f"[{p[:40]}] {f}" for f in probe_fails]

    mean_sim = sum(sims) / len(sims)
    print(f"\nMean similarity: {mean_sim:.2f} (threshold {PASS_RATIO})")
    if mean_sim < PASS_RATIO:
        failures.append(f"mean similarity {mean_sim:.2f} < {PASS_RATIO}")

    if failures:
        print(f"\nFAIL ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        print("\nDo NOT proceed to evaluation. Inspect the chat template in the "
              "GGUF (llama.cpp logs it at load) against tokenizer_config.json, "
              "the convert_hf_to_gguf warnings, and the stop-token ids in "
              "03_generate.GGUF_STOP_TOKEN_STRINGS. Fix the harness -- do not "
              "edit the HF config files or the GGUF metadata.")
        sys.exit(1)
    print(f"\nPASS: {len(PROBES)} probes, prompt tokenization identical, every "
          "turn ended on a stop token, no repetition. "
          f"llama.cpp pin: {llama_cpp_pin()}. Record this in README (Phase 0 step 5).")


if __name__ == "__main__":
    main()
