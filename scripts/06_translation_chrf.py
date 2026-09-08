"""ChrF on Flores en->X generations (free metric, matches paper's translation
scoring). Usage: python 06_translation_chrf.py --model global --precision q4_0
"""
import argparse
from collections import defaultdict

from sacrebleu.metrics import CHRF

from common import JsonlStore, load_config, result_path


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True)
    args = ap.parse_args()

    gens = JsonlStore(result_path(
        cfg, f"gen__flores__{args.model}__{args.precision}.jsonl")).load_all()
    chrf = CHRF()

    by_lang = defaultdict(lambda: ([], []))
    for g in gens:
        if g.get("reference") and g.get("response") is not None:
            hyp = g["response"].strip().split("\n")[0].strip()  # first line = translation
            by_lang[g["lang"]][0].append(hyp)
            by_lang[g["lang"]][1].append(g["reference"])

    out = JsonlStore(result_path(cfg, "chrf_scores.jsonl"))
    print(f"\nChrF — {args.model} @ {args.precision}")
    for lang in sorted(by_lang):
        hyps, refs = by_lang[lang]
        score = chrf.corpus_score(hyps, [refs]).score
        key = f"{args.model}|{args.precision}|{lang}"
        if not out.has(key):
            out.add(key, {"model": args.model, "precision": args.precision,
                          "lang": lang, "chrf": score, "n": len(hyps)})
        print(f"  {lang}: {score:5.1f}  (n={len(hyps)})")


if __name__ == "__main__":
    main()
