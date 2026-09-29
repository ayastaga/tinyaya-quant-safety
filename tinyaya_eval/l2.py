"""tinyaya_eval.l2 — Tiny Aya L2-Thinker: thinking-mode generation and judge-free metrics.

Drop this file into tinyaya_eval/ in the tinyaya-quant-safety repo and push it. All three
L2-Thinker notebooks import it; nothing is monkey-patched in a notebook.

CLI
  python -m tinyaya_eval.l2 patch                       register l2thinker* models + per-model ctx/gen
                                                        budget in config.py / prepare.py (idempotent)
  python -m tinyaya_eval.l2 probe  --precision f16      thinking-mode preflight (trigger, think ids, stop)
  python -m tinyaya_eval.l2 gen    --precision q4_0 --dataset mgsm --langs en,bn --variant L2 --n 250 --tag pilot
                                                        resumable; one process per (precision, variant, dataset)

Environment (read by config.py after `patch`): TINYAYA_MODEL=l2thinker|l2thinker_trim|l2thinker_ctrl,
TINYAYA_MAX_NEW (default 1024 -> set 8192 for MGSM/Aya, 4096 for Macaron), TINYAYA_N_CTX (set 10240).

Verified against the paper (arXiv:2609.10445) and the model card on 2026-09-28:
  * the chat template itself prepends "Think in the same language as the prompt. " to EVERY user turn and
    appends " /think" (or " /no_think" when enable_thinking=False) - the trigger is not something we add;
  * thinking delimiters: <|START_THINKING|> ... <|END_THINKING|><|START_RESPONSE|> ... <|END_RESPONSE|>;
  * Appendix D: one completion per example, 32K context; no temperature is stated. Model card: T=0.6,
    top_p=0.95. Greedy is OUR protocol (paired determinism), not the paper's.
  * tiers (Fig. 5, Common Crawl page counts): see TIER below - note th is tier 2, not 3.
  * doomlooping proxy (S3.4 / Fig. 4): 4-gram repetition score, length-normalised, averaged over 1024-token windows.
  * L2 rate: FastText LID (GlotLID fallback in the paper; FastText only here).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TRIGGER = "Think in the same language as the prompt. "
THINK = {"start": "<|START_THINKING|>", "end": "<|END_THINKING|>",
         "resp_start": "<|START_RESPONSE|>", "resp_end": "<|END_RESPONSE|>"}

# Reasoning-language variants. The template carries the L2 trigger; each variant is a controlled edit of it.
VARIANTS = {
    "L2":      "default template: L2 trigger + thinking on",
    "EN":      "trigger replaced by 'Think in English. ' (paper's En-Thinker is a different MODEL; this isolates the trigger)",
    "MIX":     "language-mixed CoT probe (KO-REAson, arXiv:2510.04230): English scaffolding, in-language entities/quotes",
    "NOTRIG":  "trigger removed, thinking on (paper's Limitations: trigger dependence)",
    "NOTHINK": "enable_thinking=False (the template writes an empty think block)",
}
VARIANT_TRIGGER = {
    "EN":  "Think in English. ",
    "MIX": "Think in English, but keep every name, quotation and key term from the prompt in the prompt's own language. ",
}

# Fig. 5 tiers (Common Crawl page counts). Only languages we may touch; extend from the paper's Table 2 if needed.
TIER = {1: "cs de en es fr id it ja nl pl pt ru tr vi zh".split(),
        2: "ar bg ca el fa fi he hu ko no ro sk sv th uk".split(),
        3: "bn et eu gl hi hr lt mr ms ne sl sr ta te ur".split(),
        4: "am cy gu ha ig jv km my sn sw tl wo xh yo zu".split()}
LANG_TIER = {l: t for t, ls in TIER.items() for l in ls}

MGSM_LANGS = "bn de en es fr ja ru sw te th zh".split()
AYA_NAMES = {"en": "english", "bn": "bengali", "th": "thai", "jv": "javanese", "de": "german", "ar": "arabic",
             "ko": "korean", "sw": "swahili", "hi": "hindi", "ta": "tamil", "te": "telugu", "ur": "urdu",
             "pa": "punjabi", "mr": "marathi", "gu": "gujarati", "ne": "nepali", "zh": "chinese", "ja": "japanese",
             "yo": "yoruba", "zu": "zulu", "am": "amharic", "id": "indonesian", "it": "italian", "tr": "turkish",
             "pt": "portuguese", "es": "spanish", "el": "greek", "tl": "tagalog", "vi": "vietnamese"}
MACARON_CSV = "https://huggingface.co/datasets/AlaaAhmed2444/Macaron/resolve/main/macaron_mcq.csv"
MACARON_ISO = {"Amharic": "am", "Brazilian Portuguese": "pt", "Chinese": "zh", "Egyptian Arabic": "ar", "Georgian": "ka",
               "Greek": "el", "Hindi": "hi", "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Kyrgyz": "ky",
               "Mexican Spanish": "es", "Moroccan Arabic": "ar", "Tagalog": "tl", "Thai": "th", "Tunisian Arabic": "ar",
               "Turkish": "tr", "Yemeni Arabic": "ar", "Yoruba": "yo", "Zulu": "zu"}
L2_45 = set("am bg bn ca cs en el eu fa fi tl ga ha he hi hu id ig it jv km lt ms mt no pa pl ru sk sw ta te th tr uk ur vi yo zh zu ar de fr ja ko".split())


# ───────────────────────────────────────────────────────────── repo patch ──
def patch_repo(repo_dir):
    """Idempotent edits so config.py knows the L2 models and reads the generation budget from the environment."""
    repo = Path(repo_dir)
    cfg = repo / "tinyaya_eval" / "config.py"
    s = cfg.read_text()
    anchor = '    "water":  {"repo": "CohereLabs/tiny-aya-water",'
    assert anchor in s, "config.py layout changed; patch by hand"
    entries = ""
    if '"l2thinker"' not in s:
        entries += ('    "l2thinker": {"repo": "CohereLabs/tiny-aya-l2-thinker",\n'
                    '                  "table7": {"min_safe": 0.0, "mean_safe": 0.0}, "table26_safe": {}, "table26_invalid": {}},\n')
    for name in ("l2thinker_trim", "l2thinker_ctrl"):
        if f'"{name}"' not in s:
            entries += (f'    "{name}": {{"repo": "CohereLabs/tiny-aya-l2-thinker", "trim_of": "l2thinker",\n'
                        f'                  "table7": {{"min_safe": 0.0, "mean_safe": 0.0}}, "table26_safe": {{}}, "table26_invalid": {{}}}},\n')
    if entries:
        s = s.replace(anchor, entries + anchor, 1)
    if "TINYAYA_MAX_NEW" not in s:
        s = s.replace("MAX_NEW_TOKENS = 1024", 'MAX_NEW_TOKENS = int(os.environ.get("TINYAYA_MAX_NEW", 1024))', 1)
        s = s.replace("N_CTX          = 4096", 'N_CTX          = int(os.environ.get("TINYAYA_N_CTX", 4096))', 1)
    cfg.write_text(s)
    prep = repo / "tinyaya_eval" / "prepare.py"
    p = prep.read_text()
    if "trim_of" not in p:
        guard = ('    if "trim_of" in config.MODELS[config.MODEL_NAME]:\n'
                 '        raise SystemExit(f"{config.MODEL_NAME} is a derived model: run vocabtrim apply / symlink, not prepare download/convert")\n')
        p = p.replace("def download():\n", "def download():\n" + guard, 1).replace("def convert():\n", "def convert():\n" + guard, 1)
        prep.write_text(p)
    return "config.py + prepare.py patched (idempotent)"


def gpu_name():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "cpu"


# ───────────────────────────────────────────────────────────── backend ──
DECODING = os.environ.get("TINYAYA_DECODING", "sample")          # "sample" (model card) | "greedy"
SAMPLING = {"temp": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0}   # model card; no min_p / top_k


def prompt_seed(item_id):
    import hashlib
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)


class L2Backend:
    """Thinking-mode wrapper around generate.Backend. Returns trace and answer separately (split on the
    <|END_THINKING|> token id, not on text: detokenize() drops special tokens)."""

    def __init__(self, precision, variant="L2", prefix=""):
        from . import config
        from .generate import Backend
        assert variant in VARIANTS, f"variant {variant!r}; expected one of {sorted(VARIANTS)}"
        self.b = Backend(precision)
        self.llm, self.cfg = self.b.llm, config
        self.variant, self.prefix = variant, prefix
        tpl = self.b.template
        assert TRIGGER in tpl, "template does not carry the L2 trigger; is this really tiny-aya-l2-thinker?"
        if variant in VARIANT_TRIGGER:
            tpl = tpl.replace(TRIGGER, VARIANT_TRIGGER[variant])
        elif variant == "NOTRIG":
            tpl = tpl.replace(TRIGGER, "")
        self.tpl = tpl
        one = lambda s: self.llm.tokenize(s.encode(), add_bos=False, special=True)
        self.ids = {k: one(v) for k, v in THINK.items()}
        bad = {k: v for k, v in self.ids.items() if len(v) != 1}
        assert not bad, f"think delimiters are not single tokens: {bad}"
        self.ids = {k: v[0] for k, v in self.ids.items()}
        self.end_think = self.ids["end"]
        self.decoding = DECODING
        self.kw = dict(self.b.kw)
        if self.decoding == "sample":
            self.kw.update(SAMPLING)
        else:
            assert self.decoding == "greedy", self.decoding
        assert self.ids["resp_end"] in self.b.stop_ids, "stop set lacks <|END_RESPONSE|>"
        assert self.end_think not in self.b.stop_ids, "stop set contains <|END_THINKING|>: answers would be cut off"

    def render(self, user_text):
        from jinja2 import Environment
        env = Environment(extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        r = env.from_string(self.tpl).render(messages=[{"role": "user", "content": user_text}], add_generation_prompt=True,
                                             bos_token=self.b.bos, eos_token="", enable_thinking=(self.variant != "NOTHINK"))
        if self.prefix and self.variant != "NOTHINK":
            assert r.rstrip().endswith(THINK["start"]), "prefix requires the prompt to end with <|START_THINKING|>"
            r = r.rstrip() + self.prefix
        return r

    def generate(self, user_text, seed=0):
        if self.decoding == "sample":
            self.llm.set_seed(int(seed))
        toks = self.llm.tokenize(self.render(user_text).encode("utf-8"), add_bos=False, special=True)
        budget = max(1, min(self.cfg.MAX_NEW_TOKENS, self.llm.n_ctx() - len(toks) - 8))
        out, finish, stop = [], "length", None
        if self.cfg.KV_RESET_PER_PROMPT:
            self.llm.reset()
        for tid in self.llm.generate(toks, **self.kw):
            if tid in self.b.stop_ids:
                finish, stop = "stop", self.b.stop_ids[tid]
                break
            out.append(int(tid))
            if len(out) >= budget:
                break
        cut = out.index(self.end_think) if self.end_think in out else None
        trace_ids = out[:cut] if cut is not None else (out if self.variant != "NOTHINK" else [])
        ans_ids = out[cut + 1:] if cut is not None else ([] if self.variant != "NOTHINK" else out)
        dec = lambda t: self.llm.detokenize(t).decode("utf-8", "replace")
        return {"trace": dec(trace_ids), "answer": dec(ans_ids), "n_prompt": len(toks), "n_trace": len(trace_ids),
                "n_answer": len(ans_ids), "n_gen": len(out), "finish": finish, "stop_token": stop,
                "truncated": finish == "length", "think_closed": cut is not None,
                "doomloop": doomloop_score(trace_ids)}

    def close(self):
        self.b.close()


# ───────────────────────────────────────────────────────────── metrics ──
def doomloop_score(ids, n=4, window=1024):
    """Paper S3.4: 4-gram repetition score over the thinking trace, length-normalised, averaged over 1024-token
    windows. Per window: sum_{g: c_g>1}(c_g-1) / sum_g c_g. Windows shorter than n tokens are skipped."""
    ids = list(ids)
    scores = []
    for s in range(0, max(len(ids), 1), window):
        w = ids[s:s + window]
        if len(w) < n:
            continue
        c = collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
        tot = sum(c.values())
        scores.append(sum(v - 1 for v in c.values() if v > 1) / tot if tot else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def doomloop_score_text(text, n=4, window=1024):
    """Whitespace-token version for re-analysis without the model; use the id version from generation rows."""
    return doomloop_score(text.split(), n, window)


_LID = None


def lid(text, root="/content"):
    """FastText lid.176 on the prose portion of a trace (digits, punctuation, code stripped). Returns iso code or None."""
    global _LID
    if _LID is None:
        import fasttext  # pip install fasttext-wheel
        p = Path(root) / "lid.176.bin"
        if not p.exists():
            subprocess.run(["wget", "-q", "-O", str(p), "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"], check=True)
        _LID = fasttext.load_model(str(p))
    prose = re.sub(r"\\[a-zA-Z]+\{[^}]*\}|\$[^$]*\$|[\d\W_]+", " ", text).strip()
    if len(prose) < 20:
        return None
    # _LID.predict() breaks under NumPy 2 (np.array(..., copy=False)); the C++ binding returns [(prob, label)] directly
    pairs = _LID.f.predict(prose[:3000].replace("\n", " "), 1, 0.0, "strict")
    return pairs[0][1].replace("__label__", "") if pairs else None


def l2_flag(row):
    """True if the trace's predominant language equals the prompt language (paper: L2 reasoning rate)."""
    lang = lid(row["trace"]) if row.get("trace") else None
    return lang == row["lang"], lang


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANS_MARK = re.compile(r"(?:final answer|answer|উত্তর|उत्तर|คำตอบ|antwort|réponse|respuesta|答案|答え|ответ|jibu|సమాధానం|விடை|جواب|الإجابة)", re.I)


def _ascii_digits(s):
    """Bengali/Devanagari/Thai/Arabic-Indic/... digits -> ASCII (models often answer in native numerals)."""
    import unicodedata
    return "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in s)


def _clean(s):
    s = _ascii_digits(s)
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}", r" \1 ", s)
    for a, b in (("\\$", ""), ("$", ""), ("\\,", ""), ("\\!", ""), ("\u202f", ""), ("\u00a0", ""), ("\u2009", ""), ("\\%", "%")):
        s = s.replace(a, b)
    return s


def _norm_num(s):
    s = _clean(str(s)).replace(",", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v == int(v) else v


def _boxed(text):
    r"""Contents of every \boxed{...}, brace-matched (handles \boxed{\$460} and \boxed{18\text{ dollars}})."""
    out, k = [], 0
    while True:
        k = text.find("\\boxed{", k)
        if k < 0:
            return out
        depth, i = 1, k + len("\\boxed{")
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0); i += 1
        out.append(text[k + len("\\boxed{"):i - 1]); k = i


def _first_num(s):
    m = _NUM.findall(_clean(s))
    return _norm_num(m[0]) if m else None


def mgsm_pred(answer):
    r"""Priority: last \boxed{} -> number after the last answer marker -> last number outside trailing
    parenthetical asides (models append '(If instead ... 120 ...)' notes after the real answer)."""
    for b in reversed(_boxed(answer)):
        v = _first_num(b)
        if v is not None:
            return v
    marks = list(_ANS_MARK.finditer(answer))
    if marks:
        v = _first_num(answer[marks[-1].end():marks[-1].end() + 120])
        if v is not None:
            return v
    body = re.sub(r"\*?\([^()]{40,}\)\*?", " ", answer)
    nums = _NUM.findall(_clean(body)) or _NUM.findall(_clean(answer))
    return _norm_num(nums[-1]) if nums else None


def mgsm_correct(answer, gold):
    p = mgsm_pred(answer)
    return p is not None and _norm_num(str(gold)) == p


def mcq_pred(answer, options):
    """Letter A-D if the answer names one; else the unique option string present; else None."""
    a = answer.strip()
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", a[:200]) or re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", a)
    if m:
        return m.group(1)
    hits = [i for i, o in enumerate(options) if str(o).strip() and str(o).strip() in a]
    return "ABCD"[hits[0]] if len(hits) == 1 else None


def mcq_correct(answer, options, gold):
    return mcq_pred(answer, options) == gold


def paired_binary(a, b, key):
    """a, b: dict id -> row. Returns (n, b_wins, c_wins, p) with exact McNemar on rows present in both."""
    from .analyze import mcnemar_exact
    ks = [k for k in a if k in b]
    x = sum(bool(a[k][key]) and not bool(b[k][key]) for k in ks)
    y = sum(bool(b[k][key]) and not bool(a[k][key]) for k in ks)
    return len(ks), x, y, mcnemar_exact(x, y)


def paired_bootstrap(a, b, key, n_boot=2000, seed=0):
    """Mean(b-a) of a numeric per-row metric with a 95% bootstrap CI over paired rows."""
    import random
    ks = [k for k in a if k in b]
    d = [float(b[k][key]) - float(a[k][key]) for k in ks]
    if not d:
        return 0.0, (0.0, 0.0), 0
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]), len(d)


# ───────────────────────────────────────────────────────────── datasets ──
def load_items(dataset, langs, n, skip=200_000, csv_dir="/content"):
    """Returns list of {id, lang, prompt, gold, aux}. gold: str(number) for mgsm, letter for macaron, None for aya."""
    items = []
    n = n or 10**9          # 0 = every item
    if dataset == "mgsm":
        from datasets import load_dataset
        for lang in langs:
            assert lang in MGSM_LANGS, f"MGSM has no {lang}; use aya for it"
            ds = load_dataset("juletxara/mgsm", lang, split="test")
            for i, r in enumerate(ds):
                if i >= n:
                    break
                items.append({"id": f"mgsm:{lang}:{i}", "lang": lang, "prompt": r["question"], "gold": str(r["answer_number"]), "aux": {}})
    elif dataset == "aya":
        from datasets import load_dataset
        for lang in langs:
            ds = load_dataset("CohereForAI/aya_collection_language_split", AYA_NAMES[lang], split="train", streaming=True)
            ok = lambda r: 30 < len((r.get("inputs") or "").strip()) < 800
            got = [r["inputs"].strip() for r, _ in zip((r for r in ds.skip(skip) if ok(r)), range(n))]
            if len(got) < n:  # small language: skip exhausted the split
                got = [r["inputs"].strip() for r, _ in zip((r for r in ds if ok(r)), range(n))]
            for i, p in enumerate(got):
                items.append({"id": f"aya:{lang}:{i}", "lang": lang, "prompt": p, "gold": None, "aux": {}})
    elif dataset in ("macaron", "macaron_en"):
        import pandas as pd
        csv = Path(csv_dir) / "macaron_mcq.csv"
        if not csv.exists():
            subprocess.run(["wget", "-q", "-O", str(csv), MACARON_CSV], check=True)
        df = pd.read_csv(csv)
        df["iso"] = df.language.map(MACARON_ISO)
        if langs:
            df = df[df.iso.isin(langs)]
        en = dataset == "macaron_en"
        for r in df.itertuples():
            opts = [r.english_option1, r.english_option2, r.english_option3, r.english_option4] if en else [r.option1, r.option2, r.option3, r.option4]
            q = r.english_question if en else r.question
            # correct_option is the option TEXT, given in English for ~74% of rows and in the local language for the
            # rest; the English and local option lists are index-aligned, so resolve the index through whichever matches.
            co = str(r.correct_option).strip()
            lopts = [str(o).strip() for o in (r.option1, r.option2, r.option3, r.option4)]
            eopts = [str(o).strip() for o in (r.english_option1, r.english_option2, r.english_option3, r.english_option4)]
            idx = eopts.index(co) if co in eopts else (lopts.index(co) if co in lopts else None)
            if idx is None:
                continue
            # The CSV lists the correct option FIRST in every row. Shuffle with a per-item seed so position carries no
            # signal; the same permutation is used for the local and English versions, so conditions stay paired.
            import random
            perm = list(range(4)); random.Random(f"macaron:{r.id}").shuffle(perm)
            opts = [opts[j] for j in perm]
            gold = "ABCD"[perm.index(idx)]
            items.append({"id": f"macaron:{r.id}", "lang": r.iso, "gold": gold,
                          "prompt": f"{q}\n\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}",
                          "aux": {"country": r.country, "language": r.language, "options": [str(o) for o in opts],
                                  "category": r.reasoning_category, "aspect": r.cultural_aspect, "template": int(r.template_id),
                                  "covered": r.iso in L2_45, "tier": LANG_TIER.get(r.iso)}})
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    return items


# ───────────────────────────────────────────────────────────── runs ──
def out_path(dataset, variant, precision, tag="pilot"):
    from . import config
    d = Path(config.ROOT) / "runs_l2" / config.MODEL_NAME / tag
    d.mkdir(parents=True, exist_ok=True)
    suffix = "" if DECODING == "greedy" else f"__{DECODING}"          # greedy keeps the original file names
    return d / f"{dataset}__{variant}__{precision}{suffix}.jsonl"


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return {}
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def gen(precision, dataset, langs, n, variant="L2", prefix="", tag="pilot", mirror_every=20):
    from . import common, config
    path = out_path(dataset, variant, precision, tag)
    done = load_rows(path)
    items = [it for it in load_items(dataset, langs, n) if it["id"] not in done]
    print(f"[{config.MODEL_NAME}/{precision}/{variant}/{dataset}] {len(done)} done, {len(items)} to go -> {path}", flush=True)
    if not items:
        return path
    be = L2Backend(precision, variant, prefix)
    rev = common.revision()
    gpu = gpu_name()
    t0, k = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            seed = prompt_seed(it["id"])
            g = be.generate(it["prompt"], seed=seed)
            row = {"decoding": DECODING, "seed": seed if DECODING == "sample" else None, "id": it["id"], "lang": it["lang"], "gold": it["gold"], "aux": it["aux"], "dataset": dataset,
                   "variant": variant, "prefix": prefix, "precision": precision, "model": config.MODEL_NAME,
                   "revision": rev, "gpu": gpu, "max_new": config.MAX_NEW_TOKENS, "n_ctx": config.N_CTX, **g}
            if dataset == "mgsm":
                row["correct"] = mgsm_correct(g["answer"], it["gold"])
            elif dataset.startswith("macaron"):
                row["pred"] = mcq_pred(g["answer"], it["aux"]["options"])
                row["correct"] = row["pred"] == it["gold"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            k += 1
            if k % mirror_every == 0:
                common.mirror(path)
                print(time.strftime("%H:%M"), f"{k}/{len(items)} {(time.time() - t0) / k:.1f}s/row", flush=True)
    common.mirror(path)
    be.close()
    return path


def probe(precision):
    be = L2Backend(precision)
    r = be.render("2+2 কত?")
    print(f"--- {precision} --- trigger in template: {TRIGGER in r} | ends with START_THINKING: {r.rstrip().endswith(THINK['start'])}")
    print("think ids:", be.ids, "| stop ids:", sorted(be.b.stop_ids.items()))
    g = be.generate("2+2 কত?", seed=1)
    print(f"decoding={be.decoding}")
    print(f"finish={g['finish']} stop={g['stop_token']} think_closed={g['think_closed']} n_trace={g['n_trace']} n_answer={g['n_answer']}")
    print("trace tail:", repr(g["trace"][-160:])); print("answer   :", repr(g["answer"][:200]))
    ok = g["finish"] == "stop" and g["think_closed"] and g["n_answer"] > 0
    same = be.generate("2+2 কত?", seed=1)["answer"] == g["answer"]
    print("deterministic:", same, "=>", "PASS" if ok and same else "FAIL")
    be.close()
    return ok and same


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patch"); p.add_argument("--repo", default="/content/tinyaya-quant-safety")
    p = sub.add_parser("probe"); p.add_argument("--precision", required=True)
    p = sub.add_parser("gen")
    p.add_argument("--precision", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--langs", default="en"); p.add_argument("--n", type=int, default=100)
    p.add_argument("--variant", default="L2", choices=sorted(VARIANTS)); p.add_argument("--prefix", default="")
    p.add_argument("--tag", default="pilot")
    a = ap.parse_args()
    if a.cmd == "patch":
        print(patch_repo(a.repo))
    elif a.cmd == "probe":
        sys.exit(0 if probe(a.precision) else 1)
    else:
        gen(a.precision, a.dataset, [l for l in a.langs.split(",") if l], a.n, a.variant, a.prefix, a.tag)


if __name__ == "__main__":
    main()
