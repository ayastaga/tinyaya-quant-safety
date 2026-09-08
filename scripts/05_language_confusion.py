"""Language confusion (Eval B). NO GPU beyond generation, NO API — free metric.

Marchisio-style line-level pass rate with fastText lid.176 on mDolly
generations. v2: scorer exposed as importable functions for reuse by
04b_score_mgsm.py (CoT adherence).

Usage: python 05_language_confusion.py --model global --precision q4_k_m
"""
import argparse
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from common import JsonlStore, load_config, result_path

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def get_lid(cfg):
    import fasttext
    path = Path(cfg["paths"]["results"]) / "lid.176.bin"
    if not path.exists():
        print("Downloading fastText LID model...")
        urllib.request.urlretrieve(FASTTEXT_URL, path)
    return fasttext.load_model(str(path))


def split_lines(text, min_chars=10):
    return [l.strip() for l in re.split(r"[\n]+", text) if len(l.strip()) >= min_chars]


def line_pass_rate(lid, text, lang, exclude_last_line=False):
    """(pass_rate, n_lines) or (None, 0). exclude_last_line: for CoT scoring,
    drop the final-answer line (a bare numeral has no language)."""
    lines = split_lines(text)
    if exclude_last_line and lines:
        lines = lines[:-1]
    if not lines:
        return None, 0
    ok = 0
    for line in lines:
        pred = lid.predict(line.replace("\n", " "))[0][0].replace("__label__", "")
        if pred == lang or (lang == "zh" and pred in ("zh", "zh-cn", "zh-tw")):
            ok += 1
    return ok / len(lines), len(lines)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True)
    args = ap.parse_args()

    lid = get_lid(cfg)
    gens = JsonlStore(result_path(
        cfg, f"gen__mdolly__{args.model}__{args.precision}.jsonl")).load_all()
    out = JsonlStore(result_path(
        cfg, f"confusion__{args.model}__{args.precision}.jsonl"))

    for g in gens:
        if g.get("response") and not out.has(g["key"]):
            rate, n = line_pass_rate(lid, g["response"], g["lang"])
            if rate is not None:
                out.add(g["key"], {"lang": g["lang"], "line_pass_rate": rate,
                                   "n_lines": n, "model": args.model,
                                   "precision": args.precision})

    per_lang = defaultdict(list)
    for rec in out.load_all():
        per_lang[rec["lang"]].append(rec["line_pass_rate"])
    print(f"\nLine-level pass rate — {args.model} @ {args.precision}")
    for lang in sorted(per_lang):
        vals = per_lang[lang]
        print(f"  {lang}: {100*sum(vals)/len(vals):5.1f}%  (n={len(vals)})")
    # fastText covers a language subset; averages over supported langs only — document.


if __name__ == "__main__":
    main()
