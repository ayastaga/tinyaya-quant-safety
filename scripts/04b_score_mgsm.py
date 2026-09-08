"""Eval D scorer (v2). NO JUDGE — fully deterministic.

Per (model, precision): reads gen__globalmgsm__*.jsonl and emits per-example
records with {correct, cot_pass_rate, truncated}, printing per-language
{accuracy, cot_adherence, truncation_rate, n}.

Extraction contract (FROZEN once validated — see tests/test_extract.py):
  1. Normalize non-Latin digits (Bengali, Devanagari, Thai, Telugu,
     Arabic-Indic, Extended Arabic-Indic) to ASCII.
  2. Strip thousands separators (comma/space/NBSP/apostrophe between digits);
     lone decimal commas become dots.
  3. Answer = LAST number in the response.
  4. Match gold with tolerance 1e-4 (gold is int in MGSM).
A brittle extractor that fails more on degraded output manufactures a fake
effect — hence step 1-3 are deliberately generous and precision-blind.

Usage: python 04b_score_mgsm.py --model global --precision q4_0 [--no-adherence]
"""
import argparse
import re
from collections import defaultdict

from common import JsonlStore, load_config, result_path

_DIGIT_BLOCKS = {  # unicode zero codepoint per script
    0x0660: "arabic-indic", 0x06F0: "ext-arabic-indic", 0x0966: "devanagari",
    0x09E6: "bengali", 0x0C66: "telugu", 0x0E50: "thai",
}
_TRANS = {}
for zero in _DIGIT_BLOCKS:
    for d in range(10):
        _TRANS[zero + d] = ord("0") + d

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_digits(text: str) -> str:
    return text.translate(_TRANS)


def clean_separators(text: str) -> str:
    # thousands separators between digit groups
    text = re.sub(r"(?<=\d)[,\u00a0' ](?=\d{3}(?:\D|$))", "", text)
    # decimal comma -> dot (e.g. "3,5" not followed by a 3-digit group)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return text


def extract_answer(text: str):
    """Last number in the response, or None."""
    if not text:
        return None
    cleaned = clean_separators(normalize_digits(text))
    matches = _NUM_RE.findall(cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def is_correct(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except (TypeError, ValueError):
        return False


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--no-adherence", action="store_true",
                    help="skip fastText CoT-adherence (e.g. quick accuracy pass)")
    args = ap.parse_args()

    gens = JsonlStore(result_path(
        cfg, f"gen__globalmgsm__{args.model}__{args.precision}.jsonl")).load_all()
    out = JsonlStore(result_path(
        cfg, f"mgsm_scores__{args.model}__{args.precision}.jsonl"))

    lid = None
    if not args.no_adherence:
        import importlib
        conf = importlib.import_module("05_language_confusion")
        lid = conf.get_lid(cfg)

    for g in gens:
        if out.has(g["key"]):
            continue
        pred = extract_answer(g.get("response", ""))
        rec = {
            "lang": g["lang"],
            "correct": is_correct(pred, g.get("gold")),
            "pred": pred, "gold": g.get("gold"),
            "truncated": bool(g.get("truncated")),
            "model": args.model, "precision": args.precision,
        }
        if lid is not None and g.get("response"):
            import importlib
            conf = importlib.import_module("05_language_confusion")
            rate, n = conf.line_pass_rate(
                lid, g["response"], g["lang"], exclude_last_line=True)
            rec["cot_pass_rate"], rec["n_cot_lines"] = rate, n
        out.add(g["key"], rec)

    agg = defaultdict(lambda: {"n": 0, "acc": 0, "trunc": 0, "adh": [], "noext": 0})
    for r in out.load_all():
        a = agg[r["lang"]]
        a["n"] += 1
        a["acc"] += int(bool(r["correct"]))
        a["trunc"] += int(bool(r["truncated"]))
        if r.get("pred") is None:
            a["noext"] += 1
        if r.get("cot_pass_rate") is not None:
            a["adh"].append(r["cot_pass_rate"])

    print(f"\nEval D — {args.model} @ {args.precision}")
    print(f"{'lang':6s} {'acc%':>6s} {'cot-adh%':>9s} {'trunc%':>7s} {'no-ext%':>8s} {'n':>5s}")
    for lang in sorted(agg):
        a = agg[lang]
        adh = 100 * sum(a["adh"]) / len(a["adh"]) if a["adh"] else float("nan")
        print(f"{lang:6s} {100*a['acc']/a['n']:6.1f} {adh:9.1f} "
              f"{100*a['trunc']/a['n']:7.1f} {100*a['noext']/a['n']:8.1f} {a['n']:5d}")
    # Report all three numbers together (framing rule 9): a truncation artifact
    # must not be sold as reasoning collapse. High no-ext% flags extractor issues.


if __name__ == "__main__":
    main()
