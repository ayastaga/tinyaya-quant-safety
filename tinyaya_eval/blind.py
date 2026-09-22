"""Blinded re-export of discordant pairs for a second annotator (L29: half-blinding).

    python -m tinyaya_eval.blind export  --langs jv,bn [--n 60] [--seed 20260922]
    python -m tinyaya_eval.blind unblind --langs jv,bn

export: reads <lang>_discordant_pairs_labeled.csv (prefers the _nllb version so the annotator gets the
        independent gloss), randomises A/B per row, writes
          <lang>_blind.csv      model,key,prompt_en,A_response,A_gloss,hand_A,B_response,B_gloss,hand_B
          <lang>_blind_key.csv  key,A_is   (kept out of the annotator's hands)
unblind: joins the annotator's hand_A/hand_B back to f16/q4_0 and writes <lang>_blind_labeled.csv with
        columns hand2_f16, hand2_q4_0 next to the original hand_f16, hand_q4_0, plus a per-pair agreement flag,
        and prints Cohen's kappa on the three-way labels.

Wire into discordant.py as `discordant export --blind` by calling `export_blind()` after the normal export.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from pathlib import Path


def _results() -> Path:
    return Path(os.environ.get("TINYAYA_ROOT", "/content/tinyaya-eval")) / "results"


def _find(lang: str) -> Path | None:
    for name in (f"{lang}_discordant_pairs_labeled_nllb.csv", f"{lang}_discordant_pairs_labeled.csv"):
        for base in (_results(), _results() / "discordant",
                     Path(os.environ.get("TINYAYA_DRIVE_DIR", "/content/drive/MyDrive/tinyaya-eval") or "x") / "results"):
            p = base / name
            if p.exists():
                return p
    return None


def _gloss(r: dict, side: str) -> str:
    return r.get(f"{side}_gloss_nllb") or r.get(f"{side}_gloss") or ""


def export_blind(langs: list[str], n: int | None, seed: int) -> None:
    rng = random.Random(seed)
    for lang in langs:
        src = _find(lang)
        if not src:
            print(f"skip {lang}: no labeled pairs file")
            continue
        rows = list(csv.DictReader(open(src, encoding="utf-8")))
        if n and n < len(rows):
            rows = rng.sample(rows, n)
        blind, key = [], []
        for r in rows:
            a_is = rng.choice(("f16", "q4_0"))
            b_is = "q4_0" if a_is == "f16" else "f16"
            blind.append({
                "model": r.get("model", ""), "key": r["key"], "prompt_en": r.get("prompt_en", ""),
                "A_response": r[f"{a_is}_response"], "A_gloss": _gloss(r, a_is), "hand_A": "",
                "B_response": r[f"{b_is}_response"], "B_gloss": _gloss(r, b_is), "hand_B": "",
            })
            key.append({"key": r["key"], "A_is": a_is})
        rng.shuffle(blind)
        out, kout = _results() / f"{lang}_blind.csv", _results() / f"{lang}_blind_key.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        for path, data in ((out, blind), (kout, key)):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
        print(f"{lang}: {len(blind)} blinded pairs -> {out}   (key: {kout} — do not give to annotator)")


def _kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def unblind(langs: list[str]) -> None:
    for lang in langs:
        bpath, kpath, src = _results() / f"{lang}_blind.csv", _results() / f"{lang}_blind_key.csv", _find(lang)
        if not (bpath.exists() and kpath.exists() and src):
            print(f"skip {lang}: need {bpath.name}, {kpath.name} and the original labeled file")
            continue
        a_is = {r["key"]: r["A_is"] for r in csv.DictReader(open(kpath, encoding="utf-8"))}
        blind = {r["key"]: r for r in csv.DictReader(open(bpath, encoding="utf-8"))}
        rows = [r for r in csv.DictReader(open(src, encoding="utf-8")) if r["key"] in blind]
        h1, h2 = [], []
        for r in rows:
            b = blind[r["key"]]
            f16_side = "A" if a_is[r["key"]] == "f16" else "B"
            q4_side = "B" if f16_side == "A" else "A"
            r["hand2_f16"], r["hand2_q4_0"] = b[f"hand_{f16_side}"].strip(), b[f"hand_{q4_side}"].strip()
            r["agree_f16"] = str(r["hand2_f16"] == r["hand_f16"].strip())
            r["agree_q4_0"] = str(r["hand2_q4_0"] == r["hand_q4_0"].strip())
            h1 += [r["hand_f16"].strip(), r["hand_q4_0"].strip()]
            h2 += [r["hand2_f16"], r["hand2_q4_0"]]
        out = _results() / f"{lang}_blind_labeled.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        agree = sum(x == y for x, y in zip(h1, h2))
        print(f"{lang}: {len(rows)} pairs, {agree}/{len(h1)} labels agree ({agree/len(h1):.1%}), "
              f"Cohen's kappa = {_kappa(h1, h2):.3f} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "unblind"])
    ap.add_argument("--langs", default="jv,bn")
    ap.add_argument("--n", type=int, default=None, help="subsample size per language (default: all)")
    ap.add_argument("--seed", type=int, default=20260922)
    a = ap.parse_args()
    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    export_blind(langs, a.n, a.seed) if a.cmd == "export" else unblind(langs)


if __name__ == "__main__":
    main()
