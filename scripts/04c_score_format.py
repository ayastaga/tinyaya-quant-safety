"""Eval E scorer (v2). NO JUDGE — jsonschema-based ladder of monotone rates.

Per (model, precision): reads gen__toolformat__*.jsonl (each record carries its
task's schema) and reports per-language:
  parse_rate         response contains exactly-parseable JSON object
  schema_conformance parses AND validates against the schema
  required_fields    parses AND all required keys present (superset of above
                     minus type checks — reported for diagnosis)
  value_types        required keys present AND types correct (== conformance
                     for our simple schemas; kept for ladder clarity)
  key_drift_rate     parsed objects containing any key NOT in the schema
                     (logs examples — Swahili key names at Q4 would be the
                     single most vivid finding on this axis)

Parsing is deliberately forgiving about wrappers (markdown fences, prose
before/after): we extract the first balanced {...} block. Being strict about
wrappers would measure formatting habits, not structural degradation.

Usage: python 04c_score_format.py --model global --precision q4_0
"""
import argparse
import json
import re
from collections import defaultdict

import jsonschema

from common import JsonlStore, load_config, result_path

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_block(text: str):
    """First balanced top-level {...} in text (fence-aware). None if absent."""
    if not text:
        return None
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def score_one(response: str, schema: dict) -> dict:
    obj = extract_json_block(response)
    rec = {"parsed": obj is not None, "conform": False,
           "required_ok": False, "types_ok": False,
           "drift_keys": []}
    if obj is None:
        return rec
    required = schema.get("required", list(schema.get("properties", {})))
    props = schema.get("properties", {})
    rec["required_ok"] = all(k in obj for k in required)
    rec["drift_keys"] = [k for k in obj if k not in props]
    try:
        jsonschema.validate(obj, schema)
        rec["conform"] = True
        rec["types_ok"] = True
    except jsonschema.ValidationError:
        # types_ok: required present and each present required value type-checks
        def type_ok(v, spec):
            t = spec.get("type")
            return (t is None
                    or (t == "string" and isinstance(v, str))
                    or (t == "number" and isinstance(v, (int, float))
                        and not isinstance(v, bool))
                    or (t == "integer" and isinstance(v, int)
                        and not isinstance(v, bool)))
        rec["types_ok"] = rec["required_ok"] and all(
            type_ok(obj[k], props.get(k, {})) for k in required if k in obj)
    return rec


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True)
    args = ap.parse_args()

    gens = JsonlStore(result_path(
        cfg, f"gen__toolformat__{args.model}__{args.precision}.jsonl")).load_all()
    out = JsonlStore(result_path(
        cfg, f"format_scores__{args.model}__{args.precision}.jsonl"))

    for g in gens:
        if out.has(g["key"]):
            continue
        rec = score_one(g.get("response", ""), g["schema"])
        out.add(g["key"], {"lang": g["lang"], "task": g.get("task"), **rec,
                           "model": args.model, "precision": args.precision})

    agg = defaultdict(lambda: defaultdict(int))
    drift_examples = defaultdict(list)
    for r in out.load_all():
        a = agg[r["lang"]]
        a["n"] += 1
        for k in ("parsed", "conform", "required_ok", "types_ok"):
            a[k] += int(bool(r[k]))
        if r["drift_keys"]:
            a["drift"] += 1
            if len(drift_examples[r["lang"]]) < 5:
                drift_examples[r["lang"]].append(r["drift_keys"])

    print(f"\nEval E — {args.model} @ {args.precision}")
    print(f"{'lang':6s} {'parse%':>7s} {'schema%':>8s} {'req%':>6s} {'types%':>7s} {'drift%':>7s} {'n':>5s}")
    for lang in sorted(agg):
        a = agg[lang]
        n = a["n"]
        print(f"{lang:6s} {100*a['parsed']/n:7.1f} {100*a['conform']/n:8.1f} "
              f"{100*a['required_ok']/n:6.1f} {100*a['types_ok']/n:7.1f} "
              f"{100*a['drift']/n:7.1f} {n:5d}")
        if drift_examples[lang]:
            print(f"       drift examples: {drift_examples[lang]}")
    # Framing rule 8: this is format robustness, not agentic capability.


if __name__ == "__main__":
    main()
