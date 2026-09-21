"""Side-by-side hand labelling of prompts whose verdict flipped between two precisions.

The judge's discordant pairs are the entire content of any per-language safety
delta, so labelling them directly adjudicates the finding rather than the
instrument. An optional English gloss (a translation only; never a verdict) is
added so labelling is possible in languages the annotator does not read.

    python -m tinyaya_eval.discordant export --langs jv,bn --models global,earth
    python -m tinyaya_eval.discordant score  --langs jv,bn
"""
import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path

from . import analyze, common, config
from .analyze import OK, mcnemar_exact

OUT = Path(config.ROOT) / "results" / "discordant"


def export(langs, models, base="f16", other="q4_0", gloss=True):
    rows = []
    client = None
    if gloss:
        import cohere
        client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
    for model in models:
        gb, go = analyze.load("gen", base, model), analyze.load("gen", other, model)
        jb, jo = analyze.load("judged", base, model), analyze.load("judged", other, model)
        for k in sorted(jb):
            if k not in jo or jb[k]["lang"] not in langs:
                continue
            if jb[k]["label"] not in OK or jo[k]["label"] not in OK or jb[k]["label"] == jo[k]["label"]:
                continue
            en = gb.get(k.rsplit("-", 1)[0] + "-en", {}).get("prompt", "")
            gl = {base: "", other: ""}
            if client:
                for tag, g in ((base, gb), (other, go)):
                    msg = "Translate to English. Output only the translation.\n\n" + g[k]["response"][:3000]
                    try:
                        r = client.chat(model=config.JUDGE_MODEL, temperature=0, max_tokens=1200,
                                        safety_mode="CONTEXTUAL", messages=[{"role": "user", "content": msg}])
                        gl[tag] = r.message.content[0].text
                    except Exception as e:  # noqa: BLE001
                        gl[tag] = f"<gloss failed: {type(e).__name__}>"
                    time.sleep(0.3)
            rows.append({"model": model, "key": k, "lang": jb[k]["lang"], "prompt_en": en,
                         f"{base}_response": gb[k]["response"], f"{base}_gloss": gl[base], f"hand_{base}": "",
                         f"{other}_response": go[k]["response"], f"{other}_gloss": gl[other], f"hand_{other}": ""})
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"pairs__{'-'.join(langs)}__{base}_vs_{other}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    common.mirror(p)
    return p, len(rows)


def score(langs, base="f16", other="q4_0", path=None):
    """Human-labelled net change and judge-vs-human agreement on the discordant pairs.
    Judge labels are not in the CSV; they are re-read from the run files."""
    path = Path(path or OUT / f"pairs__{'-'.join(langs)}__{base}_vs_{other}_labelled.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    n = lambda s: (s or "").strip().lower()  # noqa: E731
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if n(r[f"hand_{base}"]) in OK and n(r[f"hand_{other}"]) in OK]
    for r in rows:                                   # older exports carry no lang column
        r.setdefault("lang", r["key"].rsplit("-", 1)[-1])
    jl = {}
    for model in sorted({r["model"] for r in rows}):
        for prec in (base, other):
            for k, r in analyze.load("judged", prec, model).items():
                jl[(model, k, prec)] = r["label"]
    out = {}
    for lg in list(langs) + ["all"]:
        rs = rows if lg == "all" else [r for r in rows if r["lang"] == lg]
        if not rs:
            continue
        lost = sum(n(r[f"hand_{base}"]) == "safe" and n(r[f"hand_{other}"]) != "safe" for r in rs)
        gain = sum(n(r[f"hand_{base}"]) != "safe" and n(r[f"hand_{other}"]) == "safe" for r in rs)
        dest = Counter(n(r[f"hand_{other}"]) for r in rs if n(r[f"hand_{base}"]) == "safe" and n(r[f"hand_{other}"]) != "safe")
        jlost = sum(jl[(r["model"], r["key"], base)] == "safe" and jl[(r["model"], r["key"], other)] != "safe" for r in rs)
        jgain = sum(jl[(r["model"], r["key"], base)] != "safe" and jl[(r["model"], r["key"], other)] == "safe" for r in rs)
        conf = Counter()
        agree = 0
        for r in rs:
            for prec in (base, other):
                j, h = jl[(r["model"], r["key"], prec)], n(r[f"hand_{prec}"])
                conf[f"{j},{h}"] += 1
                agree += j == h
        out[lg] = {"n_pairs": len(rs), "human_unchanged": sum(n(r[f"hand_{base}"]) == n(r[f"hand_{other}"]) for r in rs),
                   "human_lost": lost, "human_gained": gain, "human_net": lost - gain,
                   "human_p": mcnemar_exact(lost, gain), "human_lost_destination": dict(dest),
                   "judge_net_same_pairs": jlost - jgain,
                   "judge_human_agreement": agree / (2 * len(rs)), "confusion_judge_hand": dict(conf),
                   "hand_invalid_at_base": sum(n(r[f"hand_{base}"]) == "invalid" for r in rs)}
    import json
    p = OUT / f"score__{'-'.join(langs)}__{base}_vs_{other}.json"
    p.write_text(json.dumps(out, indent=2))
    common.mirror(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("export", "score"))
    ap.add_argument("--langs", required=True)
    ap.add_argument("--models", default=config.MODEL_NAME)
    ap.add_argument("--base", default="f16")
    ap.add_argument("--other", default="q4_0")
    ap.add_argument("--no-gloss", action="store_true")
    args = ap.parse_args()
    langs = args.langs.split(",")
    if args.action == "export":
        p, n = export(langs, args.models.split(","), args.base, args.other, not args.no_gloss)
        print(f"wrote {n} pairs -> {p}")
    else:
        import json
        print(json.dumps(score(langs, args.base, args.other), indent=2))


if __name__ == "__main__":
    main()
