"""Aggregate all results into the tables/figures the paper is missing (v2).

Produces (in results/):
  table_safety.csv       (model,precision) x (min/mean/per-lang safe rate, n per lang)
  table_confusion.csv    same shape for line-level pass rate
  table_chrf.csv         ChrF per (model, precision, lang)
  table_reasoning.csv    (v2) acc / cot_adherence / trunc_rate / no_extract per lang
  table_format.csv       (v2) parse/schema/required/types/drift rates per lang
  fig_safety_floor.png            min safe rate vs precision, line per model
  fig_safety_by_lang.png          per-language Q4_0-BF16 deltas, sorted
  fig_reasoning_vs_precision.png  (v2) acc/adherence/trunc panels, Global
  fig_format_validity.png         (v2) schema-conformance vs precision per lang
  fig_webpresence_all_axes.png    (v2, "money plot") safety/reasoning/format
                                  Q4_0-BF16 deltas vs log web presence
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import load_config

PRECISION_ORDER = ["bf16", "q8_0", "q4_k_m", "q4_0", "q3_k_m", "q2_k"]
META = {"model", "precision", "min_safe", "mean_safe", "mean", "min"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def order(df):
    df["_p"] = df["precision"].map({p: i for i, p in enumerate(PRECISION_ORDER)})
    return df.sort_values(["model", "_p"]).drop(columns="_p")


def safety_table(rd):
    rows = []
    for path in glob.glob(f"{rd}/judged__multijail__*.jsonl"):
        recs = load_jsonl(path)
        if not recs:
            continue
        per = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["label"] in ("safe", "unsafe", "invalid"):
                per[r["lang"]][1] += 1
                per[r["lang"]][0] += (r["label"] == "safe")
        rates = {l: 100 * s / n for l, (s, n) in per.items() if n}
        ns = {f"n_{l}": n for l, (_, n) in per.items()}
        rows.append({"model": recs[0]["model"], "precision": recs[0]["precision"],
                     "min_safe": min(rates.values()),
                     "mean_safe": sum(rates.values()) / len(rates), **rates, **ns})
    return pd.DataFrame(rows)


def reasoning_table(rd):
    rows = []
    for path in glob.glob(f"{rd}/mgsm_scores__*.jsonl"):
        recs = load_jsonl(path)
        if not recs:
            continue
        agg = defaultdict(lambda: {"n": 0, "acc": 0, "tr": 0, "ne": 0, "adh": []})
        for r in recs:
            a = agg[r["lang"]]
            a["n"] += 1
            a["acc"] += bool(r["correct"])
            a["tr"] += bool(r["truncated"])
            a["ne"] += (r.get("pred") is None)
            if r.get("cot_pass_rate") is not None:
                a["adh"].append(r["cot_pass_rate"])
        for lang, a in agg.items():
            rows.append({
                "model": recs[0]["model"], "precision": recs[0]["precision"],
                "lang": lang, "n": a["n"], "acc": 100 * a["acc"] / a["n"],
                "cot_adherence": (100 * sum(a["adh"]) / len(a["adh"])) if a["adh"] else math.nan,
                "trunc_rate": 100 * a["tr"] / a["n"],
                "no_extract": 100 * a["ne"] / a["n"]})
    return pd.DataFrame(rows)


def format_table(rd):
    rows = []
    for path in glob.glob(f"{rd}/format_scores__*.jsonl"):
        recs = load_jsonl(path)
        if not recs:
            continue
        agg = defaultdict(lambda: defaultdict(int))
        for r in recs:
            a = agg[r["lang"]]
            a["n"] += 1
            for k in ("parsed", "conform", "required_ok", "types_ok"):
                a[k] += bool(r[k])
            a["drift"] += bool(r["drift_keys"])
        for lang, a in agg.items():
            n = a["n"]
            rows.append({"model": recs[0]["model"], "precision": recs[0]["precision"],
                         "lang": lang, "n": n,
                         "parse_rate": 100 * a["parsed"] / n,
                         "schema_conformance": 100 * a["conform"] / n,
                         "required_fields": 100 * a["required_ok"] / n,
                         "value_types": 100 * a["types_ok"] / n,
                         "key_drift": 100 * a["drift"] / n})
    return pd.DataFrame(rows)


def confusion_table(rd):
    rows = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(f"{rd}/confusion__*.jsonl"):
        for r in load_jsonl(path):
            rows[(r["model"], r["precision"])][r["lang"]].append(r["line_pass_rate"])
    out = []
    for (m, p), langs in rows.items():
        per = {l: 100 * sum(v) / len(v) for l, v in langs.items()}
        out.append({"model": m, "precision": p, "mean": sum(per.values()) / len(per),
                    "min": min(per.values()), **per})
    return pd.DataFrame(out)


def fig_safety_floor(df, path):
    plt.figure(figsize=(7, 4.5))
    for model, g in df.groupby("model"):
        g = g.set_index("precision").reindex([p for p in PRECISION_ORDER
                                              if p in set(g["precision"])])
        plt.plot(g.index, g["min_safe"], marker="o", label=model)
    plt.ylabel("Min safe response rate across languages (%)")
    plt.xlabel("Precision")
    plt.title("Does the safety floor survive quantization? (MultiJail)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=200)
    plt.close()


def fig_safety_by_lang(df, path, model="global"):
    g = df[df.model == model].set_index("precision")
    if not {"bf16", "q4_0"} <= set(g.index):
        return
    langs = [c for c in g.columns if c not in META and not c.startswith("n_")]
    deltas = (g.loc["q4_0", langs] - g.loc["bf16", langs]).astype(float).sort_values()
    plt.figure(figsize=(8, 4.5))
    plt.bar(deltas.index, deltas.values)
    plt.ylabel("Δ safe rate, Q4_0 − BF16 (pp)")
    plt.title(f"Per-language safety change under 4-bit ({model})")
    plt.axhline(0, color="k", lw=0.8); plt.tight_layout(); plt.savefig(path, dpi=200)
    plt.close()


def fig_reasoning(df, path, model="global"):
    g = df[df.model == model]
    if g.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, (col, title) in zip(axes, [("acc", "Accuracy (%)"),
                                       ("cot_adherence", "CoT language adherence (%)"),
                                       ("trunc_rate", "Truncation rate (%)")]):
        for lang, gl in g.groupby("lang"):
            gl = gl.set_index("precision").reindex(
                [p for p in PRECISION_ORDER if p in set(gl.index)])
            ax.plot(gl.index, gl[col], marker="o", label=lang, alpha=0.7)
        ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Eval D under quantization ({model}) — report all three, "
                 "truncation is the confound check")
    fig.tight_layout(); fig.savefig(path, dpi=200)
    plt.close()


def fig_format(df, path, model="global"):
    g = df[df.model == model]
    if g.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for lang, gl in g.groupby("lang"):
        gl = gl.set_index("precision").reindex(
            [p for p in PRECISION_ORDER if p in set(gl.index)])
        plt.plot(gl.index, gl["schema_conformance"], marker="o", label=lang, alpha=0.7)
    plt.ylabel("Schema conformance (%)"); plt.xlabel("Precision")
    plt.title(f"Eval E: structured-output validity vs precision ({model})")
    plt.legend(fontsize=7, ncol=2); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200)
    plt.close()


def fig_webpresence(cfg, safety, reasoning, fmt, path, model="global"):
    """Money plot: Q4_0-BF16 delta per axis vs log10 web presence, shared bins."""
    wp = cfg.get("web_presence", {})
    series = {}
    s = safety[safety.model == model].set_index("precision") if not safety.empty else None
    if s is not None and {"bf16", "q4_0"} <= set(s.index):
        langs = [c for c in s.columns if c not in META and not c.startswith("n_")]
        series["safety Δ"] = {l: float(s.loc["q4_0", l]) - float(s.loc["bf16", l])
                              for l in langs if l in wp}
    for name, df, col in [("reasoning Δ", reasoning, "acc"),
                          ("format Δ", fmt, "schema_conformance")]:
        if df.empty:
            continue
        g = df[df.model == model]
        piv = g.pivot_table(index="lang", columns="precision", values=col)
        if {"bf16", "q4_0"} <= set(piv.columns):
            series[name] = {l: piv.loc[l, "q4_0"] - piv.loc[l, "bf16"]
                            for l in piv.index if l in wp}
    if not series:
        return
    plt.figure(figsize=(7.5, 5))
    for name, d in series.items():
        xs = [math.log10(wp[l]) for l in d]
        plt.scatter(xs, list(d.values()), label=name, alpha=0.8)
        for l, x, y in zip(d, xs, d.values()):
            plt.annotate(l, (x, y), fontsize=6, alpha=0.6)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("log10 web presence (%) — PLACEHOLDER bins, see config note")
    plt.ylabel("Q4_0 − BF16 (pp)")
    plt.title("Do all axes fail in the same (low-resource) languages?")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=200)
    plt.close()


def main():
    cfg = load_config()
    rd = cfg["paths"]["results"]
    out = Path(rd)

    safety = safety_table(rd)
    reasoning = reasoning_table(rd)
    fmt = format_table(rd)
    conf = confusion_table(rd)

    if not safety.empty:
        order(safety).to_csv(out / "table_safety.csv", index=False)
        fig_safety_floor(safety, out / "fig_safety_floor.png")
        fig_safety_by_lang(safety, out / "fig_safety_by_lang.png")
        print(order(safety)[["model", "precision", "min_safe", "mean_safe"]]
              .to_string(index=False))
    if not reasoning.empty:
        reasoning.to_csv(out / "table_reasoning.csv", index=False)
        fig_reasoning(reasoning, out / "fig_reasoning_vs_precision.png")
    if not fmt.empty:
        fmt.to_csv(out / "table_format.csv", index=False)
        fig_format(fmt, out / "fig_format_validity.png")
    if not conf.empty:
        order(conf).to_csv(out / "table_confusion.csv", index=False)
    chrf = Path(rd) / "chrf_scores.jsonl"
    if chrf.exists():
        pd.DataFrame(load_jsonl(chrf)).to_csv(out / "table_chrf.csv", index=False)
    fig_webpresence(cfg, safety, reasoning, fmt, out / "fig_webpresence_all_axes.png")
    print(f"\nTables and figures written to {rd}/")


if __name__ == "__main__":
    main()
