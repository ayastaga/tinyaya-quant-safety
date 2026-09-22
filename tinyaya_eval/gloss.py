"""NLLB glosses for hand-labelled discordant pairs (breaks the Command A gloss circularity, L28).

    python -m tinyaya_eval.gloss run     --langs jv,bn            # writes <name>_nllb.csv
    python -m tinyaya_eval.gloss compare --langs jv,bn [--thr 0.25]  # ranks pairs by gloss disagreement

Input:  <results>/{lang}_discordant_pairs_labeled.csv  (also tries <results>/discordant/ and TINYAYA_DRIVE_DIR/results/)
Output: <results>/{lang}_discordant_pairs_labeled_nllb.csv with f16_gloss_nllb / q4_0_gloss_nllb added,
        mirrored to TINYAYA_DRIVE_DIR/results/ when set.

Why sentence segmentation: NLLB-200 is sentence-level; whole-response input collapses on the degenerate
q4_0 outputs that are exactly the pairs under review, and a bad MT gloss would masquerade as circularity.
Why dedupe: q4_0 loops repeat segments; caching by segment text cuts work several-fold.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANG_MAP = {"jv": "jav_Latn", "bn": "ben_Beng", "th": "tha_Thai", "sw": "swh_Latn",
            "ar": "arb_Arab", "ko": "kor_Hang", "vi": "vie_Latn", "zh": "zho_Hans", "it": "ita_Latn"}
RESP_COLS = ("f16_response", "q4_0_response")


def _root() -> Path:
    return Path(os.environ.get("TINYAYA_ROOT", "/content/tinyaya-eval"))


def _drive() -> Path | None:
    d = os.environ.get("TINYAYA_DRIVE_DIR", "/content/drive/MyDrive/tinyaya-eval")
    return Path(d) if d else None


def find_input(lang: str) -> Path | None:
    name = f"{lang}_discordant_pairs_labeled.csv"
    cands = [_root() / "results" / name, _root() / "results" / "discordant" / name]
    if _drive():
        cands.append(_drive() / "results" / name)
    return next((p for p in cands if p.exists()), None)


def segments(text: str, lang: str, cap: int = 400) -> list[str]:
    stops = "।?!" if lang == "bn" else ".?!"
    parts = re.split(rf"(?<=[{re.escape(stops)}])\s+|\n+", text)
    out: list[str] = []
    for p in (s.strip() for s in parts):
        if not p:
            continue
        while len(p) > cap:  # hard-wrap runaway repetition
            out.append(p[:cap])
            p = p[cap:]
        out.append(p)
    return out


def load_model():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(dev).eval()
    model.generation_config.max_length = None  # silence max_length/max_new_tokens warning
    eng = tok.convert_tokens_to_ids("eng_Latn")
    assert eng != tok.unk_token_id, "eng_Latn unknown to tokenizer — transformers version mismatch"
    return tok, model, dev, eng


def translate(segs: list[str], src_lang: str, bundle, bs: int = 16, max_new: int = 400) -> dict[str, str]:
    import torch

    tok, model, dev, eng = bundle
    tok.src_lang = src_lang
    uniq = sorted(set(segs), key=len)
    memo: dict[str, str] = {}
    with torch.no_grad():
        for i in range(0, len(uniq), bs):
            batch = uniq[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
            gen = model.generate(**enc, forced_bos_token_id=eng, max_new_tokens=max_new, num_beams=1)
            for s, g in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                memo[s] = g
            print(f"    {min(i + bs, len(uniq))}/{len(uniq)} segments", end="\r", flush=True)
    print()
    return memo


def run(langs: list[str]) -> None:
    bundle = load_model()
    for lang in langs:
        src = LANG_MAP.get(lang)
        path = find_input(lang)
        if not src or not path:
            print(f"skip {lang}: {'no NLLB code' if not src else 'input not found'}")
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        seg_map = {(i, c): segments((r.get(c) or "")[:3000], lang) for i, r in enumerate(rows) for c in RESP_COLS}
        memo = translate([s for v in seg_map.values() for s in v], src, bundle)
        for (i, c), segs in seg_map.items():
            rows[i][c.replace("response", "gloss_nllb")] = " ".join(memo.get(s, "") for s in segs)
        outs = [_root() / "results" / f"{path.stem}_nllb.csv"]
        if _drive() and (_drive() / "results").exists():
            outs.append(_drive() / "results" / f"{path.stem}_nllb.csv")
        for out in outs:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"{len(rows)} rows -> {out}")


def jaccard(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def compare(langs: list[str], thr: float) -> None:
    """Rank pairs by minimum gloss overlap; print low-overlap count and write a triage CSV."""
    for lang in langs:
        path = _root() / "results" / f"{lang}_discordant_pairs_labeled_nllb.csv"
        if not path.exists() and _drive():
            path = _drive() / "results" / path.name
        if not path.exists():
            print(f"skip {lang}: {path.name} not found (run first)")
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        for r in rows:
            r["min_gloss_overlap"] = f"{min(jaccard(r['f16_gloss'], r['f16_gloss_nllb']), jaccard(r['q4_0_gloss'], r['q4_0_gloss_nllb'])):.3f}"
            r["relabel_f16"] = r["relabel_q4_0"] = ""  # annotator fills; blank = unchanged
        rows.sort(key=lambda r: float(r["min_gloss_overlap"]))
        low = sum(1 for r in rows if float(r["min_gloss_overlap"]) < thr)
        out = path.with_name(f"{lang}_gloss_triage.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{lang}: {low}/{len(rows)} pairs with min gloss overlap < {thr}  -> {out}")
        print("   fill relabel_f16 / relabel_q4_0 for rows you would label differently from the NLLB gloss;")
        print("   count of non-blank rows = N_lang for the write-up (report as upper bound).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    ap.add_argument("--langs", default="jv,bn")
    ap.add_argument("--thr", type=float, default=0.25)
    a = ap.parse_args()
    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    run(langs) if a.cmd == "run" else compare(langs, a.thr)


if __name__ == "__main__":
    main()
