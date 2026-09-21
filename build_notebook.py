"""Build notebooks/tinyaya_quant_safety.ipynb: clones the repo, installs the package, runs the pipeline."""
import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "notebooks" / "tinyaya_quant_safety.ipynb"
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md(r"""
# Does 4-bit quantization erode Tiny Aya's multilingual safety floor?

Paired MultiJail evaluation of Tiny Aya ([arXiv:2603.11510](https://arxiv.org/abs/2603.11510)) at f16, q8_0, q4_k_m and q4_0 on the llama.cpp path, judged by Command A with the paper's Appendix C prompt. Code: [ayastaga/tinyaya-quant-safety](https://github.com/ayastaga/tinyaya-quant-safety).

**Design.** One inference stack for every precision; the f16 GGUF is the unquantized reference, so a comparison against it varies only the weights. All 315 prompts × 10 languages. Exact McNemar on discordant pairs, reported on all prompts and on untruncated pairs. Judge-free truncation and repetition axes. Human adjudication of every discordant pair in the affected languages.

**Requirements.** Colab GPU runtime (T4 or better; all precisions of one model on the same GPU). Colab Secrets: `HF_TOKEN` with the model licence accepted on Hugging Face, `CO_API_KEY` (Cohere). Optional Google Drive for a one-way results mirror with resumable runs.

**Structure.** Part 0 installs a pinned toolchain and requires one restart. Part 1 fixes the model under audit and the session state. Parts 2–8 are idempotent and resumable; each reads the environment set in Part 1.

| Part | Content | Compute |
|---|---|---|
| 0 | Toolchain | ~10 min, then restart |
| 1 | Session state, offline tests, Drive restore | seconds |
| 2 | HF weights → f16 GGUF → three quants | ~40 min |
| 3 | Preflight gate | ~3 min |
| 4 | Generation, all precisions concurrently | ~7 h (A100), ~12 h (T4) |
| 5 | Judging | ~1.5 h, CPU runtime sufficient |
| 6 | Tables, statistics, figures | seconds |
| 7 | Judge validation and discordant-pair adjudication | annotator time |
| 8 | Export | seconds |
""")

# ── Part 0 ──────────────────────────────────────────────────────────────────
md("## 0 · Toolchain\n\nFails cheaply on missing prerequisites before any download. Versions are pinned to those recorded in the published runs.")

code(r"""
REPO_URL = "https://github.com/ayastaga/tinyaya-quant-safety"
REPO_REF = "main"            # pin to a tag for a citable run
REPO_DIR = "/content/tinyaya-quant-safety"

import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                   capture_output=True, text=True)
assert r.returncode == 0 and r.stdout.strip(), "GPU runtime required"
print(r.stdout.strip())
""")

code(r"""
import os
from google.colab import userdata
for name in ("HF_TOKEN", "CO_API_KEY"):
    os.environ[name] = userdata.get(name)

from huggingface_hub import hf_hub_download
hf_hub_download("CohereLabs/tiny-aya-global", "config.json", token=os.environ["HF_TOKEN"])
print("HF: gated access confirmed")
""")

code(r"""
import os, subprocess, torch
os.path.exists(REPO_DIR) or subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
!cd {REPO_DIR} && git fetch -q origin && git checkout -q {REPO_REF} && echo "repo @ $(git rev-parse --short HEAD)"

LLAMA_CPP_PIN = subprocess.run(["python", "-c", "import re,sys;print(re.search(r'LLAMA_CPP_PIN\s*=\s*\"(\w+)\"', open(sys.argv[1]).read()).group(1))",
                                f"{REPO_DIR}/tinyaya_eval/config.py"], capture_output=True, text=True).stdout.strip()
os.path.exists("/content/llama.cpp/.git") or subprocess.run(["git", "clone", "-q", "https://github.com/ggml-org/llama.cpp", "/content/llama.cpp"], check=True)
!cd /content/llama.cpp && git fetch -q origin && git checkout -q {LLAMA_CPP_PIN} && echo "llama.cpp @ $(git rev-parse --short HEAD)"
!cd /content/llama.cpp && ( [ -f requirements/requirements-convert_hf_to_gguf.txt ] && pip install -q -r requirements/requirements-convert_hf_to_gguf.txt || pip install -q -r requirements.txt )
!pip install -q -e /content/llama.cpp/gguf-py

cuda = (torch.version.cuda or "12.4").split(".")
cu = f"cu{cuda[0]}{cuda[1]}" if f"cu{cuda[0]}{cuda[1]}" in ("cu121", "cu122", "cu123", "cu124") else "cu124"
!pip install -q llama-cpp-python==0.3.35 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/{cu}
!pip install -q -e {REPO_DIR}
!pip install -q -U "protobuf>=5.29,<6"
print("install complete")
""")

code(r"""
import llama_cpp, os, cohere
assert llama_cpp.__version__ == "0.3.35", llama_cpp.__version__
if not llama_cpp.llama_supports_gpu_offload():
    !CMAKE_ARGS="-DGGML_CUDA=on" pip install -q --force-reinstall --no-cache-dir llama-cpp-python==0.3.35
    raise SystemExit("CUDA wheel rebuilt from source; restart the session and re-run this cell")
r = cohere.ClientV2(api_key=os.environ["CO_API_KEY"]).chat(
    model="command-a-03-2025", messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    max_tokens=4, temperature=0.0, safety_mode="CONTEXTUAL")
print("GPU offload: OK | Cohere:", r.message.content[0].text.strip())
""")

md("> **Restart the session** (Runtime → Restart session) before Part 1. Part 0 is not re-run afterwards.")

# ── Part 1 ──────────────────────────────────────────────────────────────────
md("## 1 · Session state\n\nThe model under audit and the data root are set once, as environment variables read by every module. Switching models is a change to `MODEL` here, not to code; run directories are content-addressed and cannot collide across models.")

code(r"""
MODEL      = "global"                                   # global | earth | fire | water
DATA_ROOT  = "/content/tinyaya-eval"
DRIVE_DIR  = "/content/drive/MyDrive/tinyaya-eval"      # "" disables the mirror

import os
from google.colab import userdata
os.environ.update({"TINYAYA_MODEL": MODEL, "TINYAYA_ROOT": DATA_ROOT, "TINYAYA_DRIVE_DIR": DRIVE_DIR,
                   "HF_TOKEN": userdata.get("HF_TOKEN"), "CO_API_KEY": userdata.get("CO_API_KEY")})
os.makedirs(f"{DATA_ROOT}/logs", exist_ok=True)

import importlib, tinyaya_eval.config as config, tinyaya_eval.common as common
importlib.reload(config); importlib.reload(common)
!python -m tinyaya_eval.selftest
print(f"model={config.MODEL_NAME} repo={config.HF_REPO} root={config.ROOT}")
""")

code(r"""
if DRIVE_DIR:
    from google.colab import drive
    drive.mount("/content/drive")
    print(f"restored {common.restore_runs()} file(s) from the mirror")
""")

# ── Part 2 ──────────────────────────────────────────────────────────────────
md("## 2 · Artifacts\n\nHF safetensors → f16 GGUF → q8_0 / q4_k_m / q4_0 with the pinned llama.cpp, per paper §6. Expected sizes: 6.71 / 3.57 / 2.14 / 2.03 GB. The tensor map shows `token_embd` at Q6_K in both 4-bit formats.")

code(r"""
!cd /content/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > /dev/null 2>&1 && cmake --build build --target llama-quantize -j$(nproc) 2>&1 | tail -1
!python -m tinyaya_eval.prepare download
!python -m tinyaya_eval.prepare convert
!python -m tinyaya_eval.prepare quantize
!python -m tinyaya_eval.prepare inventory
""")

# ── Part 3 ──────────────────────────────────────────────────────────────────
md("## 3 · Preflight gate\n\nFor every GGUF: the chat template renders, the stop set contains `<|END_RESPONSE|>` (261001), every probe ends with `finish=stop`, and three greedy runs of each probe are identical. Generation does not proceed past a failure.")

code(r"""
import subprocess, sys
fails = []
for prec in config.PRECISIONS:
    r = subprocess.run([sys.executable, "-m", "tinyaya_eval.generate", "--precision", prec, "--probe"],
                       capture_output=True, text=True, env=os.environ.copy())
    print(r.stdout, end="")
    if r.returncode != 0:
        fails.append(prec); print("  stderr:", r.stderr[-1500:])
assert not fails, f"preflight failed: {fails}"
""")

# ── Part 4 ──────────────────────────────────────────────────────────────────
md("## 4 · Generation\n\nGreedy decoding through llama.cpp, KV cache cleared per prompt, all precisions as concurrent single-stream processes. Resumable: re-running continues each file. `PROCS_PER_PRECISION = 2` splits languages 5/5 (≈29 GB GPU memory).")

code(r"""
N_PER_LANG          = 315        # 50 → 150 → 315 nested ladder; rows are never regenerated
PROCS_PER_PRECISION = 1          # 2 on a 40 GB GPU
!python -m tinyaya_eval.parallel --n-per-lang {N_PER_LANG} --procs-per-precision {PROCS_PER_PRECISION}
""")

# ── Part 5 ──────────────────────────────────────────────────────────────────
md("## 5 · Judging\n\nCommand A, contextual safety mode, Appendix C verbatim, one thread per precision. Replies the judge did not label are stored as `unparsed`, excluded from every rate, and re-judged by the second pass. Zero `unsafe` verdicts across all precisions is the signature of a label-parsing fault, not a result.")

code(r"""
!python -m tinyaya_eval.judge --all --workers 4
!python -m tinyaya_eval.judge --all --workers 4 --drop-unparsed
""")

code(r"""
import importlib, tinyaya_eval.analyze as analyze
from collections import Counter
importlib.reload(analyze)
total_unsafe = 0
for prec in config.PRECISIONS:
    c = Counter(r["label"] for r in analyze.load("judged", prec).values())
    total_unsafe += c["unsafe"]
    print(f"{prec:7s} safe={c['safe']:5d} unsafe={c['unsafe']:4d} invalid={c['invalid']:4d} "
          f"excluded={c['unparsed'] + c['judge_error']}")
assert total_unsafe > 0
""")

# ── Part 6 ──────────────────────────────────────────────────────────────────
md("## 6 · Results\n\nDenominator is safe + unsafe + invalid. Paired tests use only prompts with a valid verdict at both precisions. The q8_0 contrast is a control: it must sit inside noise and label churn must be monotone in quantization aggressiveness.")

code(r"""
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
df = analyze.safety_table()
print(df[["precision", "n", "safe", "unsafe", "invalid", "excluded", "min_safe", "min_lang", "mean_safe"]].round(1).to_string(index=False))
print(f"\npaper Table 7, Tiny Aya {config.MODEL_NAME.title()}: min {config.PAPER_TABLE7['min_safe']} / mean {config.PAPER_TABLE7['mean_safe']}")
per = df.set_index("precision")[[f"safe_{lg}" for lg in config.LANGS]].round(1); per.columns = config.LANGS
if config.PAPER_TABLE26_SAFE:
    per.loc["paper T26"] = [config.PAPER_TABLE26_SAFE.get(lg) for lg in config.LANGS]
print("\nsafe rate by language (%)"); print(per.to_string())
""")

code(r"""
pt = analyze.paired_table()
print(pt[["precision", "scope", "n", "safe_to_not", "safe_to_invalid", "not_to_safe", "net", "discordant", "z", "p", "significant"]]
      .round(3).to_string(index=False))
print(f"\nBonferroni alpha = {pt['bonferroni_alpha'].iloc[0]:.4f} over {len(pt)} tests")
ok, msg = analyze.control_check(); print(msg)
print("\nq4_0 by language:")
print(analyze.paired_by_language("q4_0")[["lang", "n", "safe_to_not", "safe_to_invalid", "not_to_safe", "net", "p"]].round(3).to_string(index=False))
""")

code(r"""
q = analyze.quality_table()
for col, title in (("trunc_pct", "truncation (%)"), ("repeat_ratio", "repeated 12-char n-gram ratio"),
                   ("identical_to_ref_pct", "responses byte-identical to f16 (%)")):
    print(f"\n{title}")
    print(q.pivot(index="lang", columns="precision", values=col).reindex(columns=config.PRECISIONS).round(3).to_string())
""")

code(r"""
import matplotlib.pyplot as plt
for p in analyze.figures():
    print(p)
plt.show()
""")

# ── Part 7 ──────────────────────────────────────────────────────────────────
md("""## 7 · Judge validation

7.1 draws a stratified random sample (non-safe verdicts oversampled) to `results/<model>/handlabel_sample.csv`; annotated as `handlabel_sample_labelled.csv` with `safe` / `unsafe` / `invalid` in `hand_label`, 7.2 reports per-class precision and recall of the judge.

7.3 exports every f16→q4_0 discordant pair in the named languages across the named models, side by side with an English gloss (translation only), to `results/discordant/`; annotated as `*_labelled.csv` in `hand_f16` and `hand_q4_0` without reference to judge labels, 7.4 reports the human-verified net change, where lost refusals went, and judge–human agreement on the pairs the judge flipped.""")

code(r"""
p, n = analyze.handlabel_sample(n_per_class=15)
print(n, "rows ->", p)
""")

code(r"""
import json
try:
    s = analyze.handlabel_score()
    print(json.dumps({k: v for k, v in s.items() if k != "confusion_judge_hand"}, indent=2))
    print("confusion (judge,hand):", s["confusion_judge_hand"])
except FileNotFoundError as e:
    print(e, "\njudge unvalidated for this model")
""")

code(r"""
DISCORDANT_LANGS  = "jv,bn"
DISCORDANT_MODELS = "global,earth"       # models with completed runs in the data root
!python -m tinyaya_eval.discordant export --langs {DISCORDANT_LANGS} --models {DISCORDANT_MODELS}
""")

code(r"""
!python -m tinyaya_eval.discordant score --langs {DISCORDANT_LANGS}
""")

# ── Part 8 ──────────────────────────────────────────────────────────────────
md("## 8 · Export\n\nManifest, tables, figures, discordant-pair files, run JSONL and logs for the current model, zipped and mirrored.")

code(r"""
!python -m tinyaya_eval.export
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"gpuType": "A100", "provenance": [], "name": "tinyaya_quant_safety.ipynb"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {OUT} ({len(cells)} cells)")
