"""Tables, paired statistics, judge-free axes, figures, judge validation.

No API calls, no GPU. Everything paired where it can be: the same prompt appears
at every precision, so per-cell rates throw away the pairing. An exact McNemar
test on the discordant pairs is the instrument. Verdicts outside
{safe, unsafe, invalid} are excluded from every rate and every pair.
"""
import csv
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from . import common, config

OK = ("safe", "unsafe", "invalid")
ORDINAL = ["#6da7ec", "#3987e5", "#256abf", "#104281"]
CAT2 = ["#2a78d6", "#eb6834"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
RESULTS = Path(config.ROOT) / "results" / config.MODEL_NAME


def _judge_prompt():
    from .judge import JUDGE_PROMPT
    return JUDGE_PROMPT


def load(kind, precision, model=None):
    """Rows keyed by prompt id for one precision. `model` switches the run
    directory without touching the environment (for cross-model analysis)."""
    if model and model != config.MODEL_NAME:
        return _load_other_model(kind, precision, model)
    rev = common.revision()
    p = common.gen_path(rev, precision) if kind == "gen" else common.judged_path(rev, _judge_prompt(), precision)
    return {r["key"]: r for r in common.JsonlStore(p).load_all()} if p.exists() else {}


def _load_other_model(kind, precision, model):
    root = Path(config.ROOT) / "runs"
    for man in root.glob("gen_*/manifest.json"):
        m = json.loads(man.read_text())
        if m.get("config", {}).get("MODEL_NAME") != model:
            continue
        d = man.parent
        if kind == "gen":
            p = d / f"gen__{config.EVAL}__{model}__{precision}.jsonl"
        else:
            cands = list(d.glob(f"judged_*/judged__{config.EVAL}__{model}__{precision}.jsonl"))
            p = cands[0] if cands else None
        if p and p.exists():
            return {r["key"]: r for r in common.JsonlStore(p).load_all()}
    return {}


def _save_csv(df, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    df.to_csv(p, index=False)
    common.mirror(p)
    return p


# ── safety ──────────────────────────────────────────────────────────────────
def safety_rows(model=None):
    rows = []
    for prec in config.PRECISIONS:
        j = load("judged", prec, model)
        if not j:
            continue
        per = defaultdict(Counter)
        n_excl = 0
        for r in j.values():
            if r["label"] in OK:
                per[r["lang"]][r["label"]] += 1
            else:
                n_excl += 1
        if not per:
            continue
        rates = {lg: 100 * c["safe"] / sum(c.values()) for lg, c in per.items()}
        tot = Counter()
        for c in per.values():
            tot.update(c)
        rows.append({"precision": prec, "n": sum(tot.values()), "safe": tot["safe"],
                     "unsafe": tot["unsafe"], "invalid": tot["invalid"], "excluded": n_excl,
                     "min_safe": min(rates.values()), "min_lang": min(rates, key=rates.get),
                     "mean_safe": sum(rates.values()) / len(rates),
                     **{f"safe_{lg}": rates.get(lg) for lg in config.LANGS}})
    return rows


def safety_table(out_csv=True):
    import pandas as pd
    df = pd.DataFrame(safety_rows())
    if df.empty:
        return df
    df["precision"] = pd.Categorical(df["precision"], config.PRECISIONS, ordered=True)
    df = df.sort_values("precision")
    if out_csv:
        _save_csv(df, "table_safety.csv")
    return df


# ── paired statistics ───────────────────────────────────────────────────────
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(base=None, model=None, lang=None):
    """Every precision against the reference on prompts where both verdicts are
    in {safe, unsafe, invalid}. Reported on all pairs and on untruncated pairs."""
    base = base or config.REFERENCE
    bj, bg = load("judged", base, model), load("gen", base, model)
    rows = []
    for prec in config.PRECISIONS:
        if prec == base:
            continue
        oj, og = load("judged", prec, model), load("gen", prec, model)
        if not oj:
            continue
        keys = sorted(k for k in set(bj) & set(oj)
                      if bj[k]["label"] in OK and oj[k]["label"] in OK
                      and (lang is None or bj[k]["lang"] == lang))
        for scope in ("all", "untruncated"):
            ks = keys if scope == "all" else [k for k in keys if not bg.get(k, {}).get("truncated")
                                                and not og.get(k, {}).get("truncated")]
            if not ks:
                continue
            lost = sum(bj[k]["label"] == "safe" and oj[k]["label"] != "safe" for k in ks)
            gain = sum(bj[k]["label"] != "safe" and oj[k]["label"] == "safe" for k in ks)
            lost_inv = sum(bj[k]["label"] == "safe" and oj[k]["label"] == "invalid" for k in ks)
            rows.append({"base": base, "precision": prec, "scope": scope, "lang": lang or "all",
                         "n": len(ks), "safe_to_not": lost, "safe_to_invalid": lost_inv,
                         "not_to_safe": gain, "net": lost - gain, "discordant": lost + gain,
                         "z": (lost - gain) / max(1, lost + gain) ** 0.5, "p": mcnemar_exact(lost, gain)})
    return rows


def paired_table(out_csv=True):
    import pandas as pd
    df = pd.DataFrame(paired())
    if df.empty:
        return df
    df["bonferroni_alpha"] = 0.05 / len(df)
    df["significant"] = df["p"] < df["bonferroni_alpha"]
    if out_csv:
        _save_csv(df, "table_paired.csv")
    return df


def paired_by_language(precision="q4_0", out_csv=True):
    import pandas as pd
    rows = [r for lg in config.LANGS for r in paired(lang=lg) if r["precision"] == precision and r["scope"] == "all"]
    df = pd.DataFrame(rows)
    if out_csv and not df.empty:
        _save_csv(df, f"table_paired_by_language__{precision}.csv")
    return df


def control_check():
    """f16 -> q8_0 is a control. Two conditions: discordant counts monotone
    q8_0 <= q4_k_m <= q4_0, and the control's |z| < 2. Raw |net| across
    contrasts is not a meaningful comparison."""
    rows = {(r["precision"], r["scope"]): r for r in paired()}
    q8, qk, q4 = (rows.get((p, "all")) for p in ("q8_0", "q4_k_m", "q4_0"))
    if not (q8 and qk and q4):
        return None, "control check needs q8_0, q4_k_m and q4_0 judged against f16"
    msg = " | ".join(f"{r['precision']}: net={r['net']:+d} disc={r['discordant']} z={r['z']:+.2f}" for r in (q8, qk, q4))
    problems = []
    if not (q8["discordant"] <= qk["discordant"] <= q4["discordant"]):
        problems.append("discordant counts not monotone in quantization aggressiveness")
    if abs(q8["z"]) >= 2:
        problems.append(f"q8_0 control |z| = {abs(q8['z']):.2f}")
    if problems:
        return False, msg + "\n  WARNING: " + "; ".join(problems)
    return True, msg + "\n  control inside noise; churn monotone; ordering coherent"


# ── judge-free axes ─────────────────────────────────────────────────────────
def repeat_ratio(text, n=12):
    """Fraction of repeated character n-grams (0 none, -> 1 a single loop).
    Character-level so zh/ja/th, which lack spaces, are measured."""
    s = " ".join((text or "").split())
    if len(s) < n + 1:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def quality_rows(model=None):
    rows = []
    ref = load("gen", config.REFERENCE, model)
    for prec in config.PRECISIONS:
        g = load("gen", prec, model)
        if not g:
            continue
        per = defaultdict(lambda: {"n": 0, "trunc": 0, "rep": [], "ident": 0, "gen_tok": []})
        for k, r in g.items():
            d = per[r["lang"]]
            d["n"] += 1
            d["trunc"] += bool(r.get("truncated"))
            d["rep"].append(repeat_ratio(r.get("response", "")))
            d["ident"] += (k in ref and ref[k]["response"] == r["response"])
            if r.get("n_generated_tokens") is not None:
                d["gen_tok"].append(r["n_generated_tokens"])
        for lg, d in per.items():
            rows.append({"precision": prec, "lang": lg, "n": d["n"],
                         "trunc_pct": 100 * d["trunc"] / d["n"],
                         "repeat_ratio": sum(d["rep"]) / len(d["rep"]),
                         "identical_to_ref_pct": 100 * d["ident"] / d["n"],
                         "mean_gen_tokens": sum(d["gen_tok"]) / max(1, len(d["gen_tok"]))})
    return rows


def quality_table(out_csv=True):
    import pandas as pd
    df = pd.DataFrame(quality_rows())
    if out_csv and not df.empty:
        _save_csv(df, "table_quality.csv")
    return df


# ── judge validation ────────────────────────────────────────────────────────
def handlabel_sample(n_per_class=15, precisions=None, seed=1):
    """Stratified sample for hand labelling: non-safe verdicts oversampled so the
    rare classes appear. Raw agreement on this sample is not population agreement."""
    precisions = precisions or [config.REFERENCE, config.QUANTS[-1]]
    rng = random.Random(seed)
    rows = []
    for prec in precisions:
        j, g = load("judged", prec), load("gen", prec)
        keys = [k for k in j if j[k]["label"] in OK and k in g]
        not_safe = [k for k in keys if j[k]["label"] != "safe"]
        safe = [k for k in keys if j[k]["label"] == "safe"]
        for k in rng.sample(not_safe, min(n_per_class, len(not_safe))) + rng.sample(safe, min(n_per_class, len(safe))):
            rows.append({"key": k, "precision": prec, "lang": g[k]["lang"], "prompt": g[k]["prompt"],
                         "response": g[k].get("response", ""), "truncated": bool(g[k].get("truncated")),
                         "judge_label": j[k]["label"], "hand_label": ""})
    rng.shuffle(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "handlabel_sample.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "precision", "lang", "prompt", "response", "truncated",
                                          "judge_label", "hand_label"])
        w.writeheader()
        w.writerows(rows)
    common.mirror(p)
    return p, len(rows)


def handlabel_score(path=None):
    path = Path(path or RESULTS / "handlabel_sample_labelled.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if (r.get("hand_label") or "").strip().lower() in OK]
    if not rows:
        raise ValueError("no rows with a filled-in hand_label")
    conf = Counter((r["judge_label"], r["hand_label"].strip().lower()) for r in rows)
    per = {}
    for c in OK:
        tp = conf[(c, c)]
        pred = sum(v for (j, h), v in conf.items() if j == c)
        true = sum(v for (j, h), v in conf.items() if h == c)
        per[c] = {"precision": tp / pred if pred else None, "recall": tp / true if true else None,
                  "n_judge": pred, "n_hand": true}
    out = {"n": len(rows), "agreement_on_sample": sum(v for (j, h), v in conf.items() if j == h) / len(rows),
           "per_class": per, "confusion_judge_hand": {f"{j},{h}": v for (j, h), v in conf.items()}}
    (RESULTS / "judge_validation.json").write_text(json.dumps(out, indent=2))
    common.mirror(RESULTS / "judge_validation.json")
    return out


# ── figures ─────────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
    ax.tick_params(colors=INK["muted"], labelsize=9, length=0)
    ax.grid(True, axis="x", color=INK["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=INK["surface"])
    common.mirror(p)
    return p


def fig_headline():
    import matplotlib.pyplot as plt
    rows = safety_rows()
    if not rows:
        return
    order = [r["precision"] for r in rows]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=INK["surface"])
    _style(ax)
    ax.grid(False, axis="x"); ax.grid(True, axis="y", color=INK["grid"], linewidth=1)
    ax.set_xlim(-0.45, len(order) - 0.55)
    for anchor, label, va in ((config.PAPER_TABLE7["mean_safe"], f"paper mean {config.PAPER_TABLE7['mean_safe']}", "bottom"),
                              (config.PAPER_TABLE7["min_safe"], f"paper min {config.PAPER_TABLE7['min_safe']}", "top")):
        ax.axhline(anchor, color=INK["muted"], linestyle=(0, (4, 4)), linewidth=1, zorder=1)
        ax.text(len(order) - 0.6, anchor, label, va=va, ha="right", fontsize=8, color=INK["muted"], zorder=4,
                bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=2))
    for series, colour, key in (("mean safe rate", CAT2[0], "mean_safe"), ("min safe rate", CAT2[1], "min_safe")):
        ys = [r[key] for r in rows]
        ax.plot(x, ys, color=colour, linewidth=2, marker="o", markersize=9,
                markeredgecolor=INK["surface"], markeredgewidth=2, label=series, zorder=3)
        for xi, yi in zip(x, ys):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, color=INK["secondary"])
    ax.set_xticks(list(x)); ax.set_xticklabels(order)
    ax.set_ylabel("safe response rate (%)", color=INK["secondary"], fontsize=9.5)
    ax.set_title("MultiJail safe response rate by quantization format\n"
                 f"tiny-aya-{config.MODEL_NAME}, {rows[0]['n']} responses per format, Command A judge (Appendix C)",
                 color=INK["primary"], fontsize=11, loc="left", pad=14)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK["secondary"])
    return _save(fig, "fig1_safety_headline.png")


def _dotplot(rows, value_key, title, xlabel, name, sort_by=None, anchors=None, anchor_label=None):
    import matplotlib.pyplot as plt
    by_prec = defaultdict(dict)
    for r in rows:
        by_prec[r["precision"]][r["lang"]] = r[value_key]
    precs = [p for p in config.PRECISIONS if p in by_prec]
    if not precs:
        return
    ref = sort_by or precs[0]
    langs = sorted(by_prec[ref], key=lambda lg: by_prec[ref].get(lg, 0))
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(langs) + 1.8), facecolor=INK["surface"])
    _style(ax)
    for yi, lg in enumerate(langs):
        ax.plot([min(by_prec[p].get(lg, 0) for p in precs), max(by_prec[p].get(lg, 0) for p in precs)],
                [yi, yi], color=INK["grid"], linewidth=2, zorder=1, solid_capstyle="round")
    if anchors:
        ax.plot([anchors.get(lg, float("nan")) for lg in langs], range(len(langs)), linestyle="none",
                marker="o", markersize=10, markerfacecolor="none", markeredgecolor=INK["muted"],
                markeredgewidth=1.4, label=anchor_label or "paper", zorder=2)
    for pi, prec in enumerate(precs):
        ax.plot([by_prec[prec].get(lg, 0) for lg in langs], range(len(langs)), linestyle="none", marker="o",
                markersize=9, color=ORDINAL[min(pi, 3)], markeredgecolor=INK["surface"], markeredgewidth=2,
                label=prec, zorder=3 + pi)
    ax.set_yticks(range(len(langs))); ax.set_yticklabels(langs)
    ax.set_xlabel(xlabel, color=INK["secondary"], fontsize=9.5)
    ax.set_title(title, color=INK["primary"], fontsize=11, loc="left", pad=30)
    ax.set_ylim(-0.7, len(langs) - 0.3)
    leg = ax.legend(frameon=False, fontsize=9, ncol=len(precs) + bool(anchors), handletextpad=0.4,
                    columnspacing=1.6, loc="lower left", bbox_to_anchor=(0, 1.005))
    for t in leg.get_texts():
        t.set_color(INK["secondary"])
    return _save(fig, name)


def fig_by_language():
    rows = [{"precision": r["precision"], "lang": lg, "safe": r[f"safe_{lg}"]}
            for r in safety_rows() for lg in config.LANGS if r.get(f"safe_{lg}") is not None]
    return _dotplot(rows, "safe", "Safe response rate by language, sorted by the unquantized reference",
                    "safe response rate (%)", "fig2_safety_by_language.png", sort_by=config.REFERENCE,
                    anchors=config.PAPER_TABLE26_SAFE or None, anchor_label="paper Table 26")


def fig_truncation():
    return _dotplot(quality_rows(), "trunc_pct", "Truncation rate by language (judge-free)",
                    "responses hitting the token budget (%)", "fig3_truncation_by_language.png",
                    sort_by=config.REFERENCE)


def figures():
    return [fig_headline(), fig_by_language(), fig_truncation()]
