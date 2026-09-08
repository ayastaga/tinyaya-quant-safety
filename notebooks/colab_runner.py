# %% [markdown]
# # Tiny Aya Quantized-Safety Audit — Colab runner
# Paste cells into Colab (or open as notebook via jupytext). Runtime: T4 is
# enough for GGUF cells; switch to A100/L4 for BF16 and QLoRA cells.

# %% Setup
# !git clone https://github.com/<you>/tinyaya-quant-safety && cd tinyaya-quant-safety
# %cd tinyaya-quant-safety
# !pip -q install -r requirements.txt
# !CMAKE_ARGS="-DGGML_CUDA=on" pip -q install llama-cpp-python --no-cache-dir
import os
from google.colab import userdata  # noqa

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["CO_API_KEY"] = userdata.get("CO_API_KEY")

# %% Day 1 — download + quantize + inspect (no GPU needed; T4 runtime fine)
# !python scripts/00_download_models.py --models global
# !bash scripts/01_quantize.sh global
# !python scripts/02_inspect_tensor_map.py models/gguf/global-*.gguf
# -> READ THE TENSOR MAP OUTPUT NOW. It shapes the whole mechanism story.

# %% Day 1-2 — first signal: Global, MultiJail, 50/lang, all precisions
# for prec in ["q8_0", "q4_k_m", "q4_0"]:
#     !python scripts/03_generate.py --model global --precision {prec} --eval multijail --max-per-lang 50
# switch runtime to A100 for the BF16 baseline:
# !python scripts/03_generate.py --model global --precision bf16 --eval multijail --max-per-lang 50

# %% Judge + first look
# for prec in ["bf16", "q8_0", "q4_k_m", "q4_0"]:
#     !python scripts/04_judge_multijail.py --model global --precision {prec}
# !python scripts/07_aggregate.py
# SANITY GATE: BF16 global mean/min should be near paper's 91.1/87.0.
# If not, fix the judge prompt / chat template BEFORE trusting any deltas.

# %% Scale out (order of value): earth -> confusion(mdolly) -> flores -> fire/water
# !python scripts/00_download_models.py --models earth
# !bash scripts/01_quantize.sh earth
# ... repeat 03/04 for earth; then:
# for prec in ["bf16", "q4_0"]:
#     !python scripts/03_generate.py --model global --precision {prec} --eval mdolly --max-per-lang 30
#     !python scripts/05_language_confusion.py --model global --precision {prec}

# %% Backup results to Drive every session (Colab storage is ephemeral!)
# from google.colab import drive
# drive.mount('/content/drive')
# !rsync -av results/ /content/drive/MyDrive/tinyaya-audit-results/

# %% v2 — Phase 0 blocking gate (A100/L4: loads BF16 + Q8_0)
# !python scripts/02b_template_check.py --model global

# %% v2 — Phase 0.5 viability gates (BF16 only, tiny scale)
# !python scripts/03_generate.py --model global --precision bf16 --eval globalmgsm --max-per-lang 20
# !python scripts/04b_score_mgsm.py --model global --precision bf16
# !CO_API_KEY=... python scripts/10_build_toolformat_prompts.py --per-lang 60   # one-time
# !python scripts/03_generate.py --model global --precision bf16 --eval toolformat --max-per-lang 20
# !python scripts/04c_score_format.py --model global --precision bf16
# Record gate outcomes in README. Eval E: <30% schema conformance at rung (ii)? drop rung / cut axis.

# %% v2 — Eval D full pass while the judge queue drains (T4 ok for GGUF)
# for prec in ["q8_0", "q4_k_m", "q4_0"]:
#     !python scripts/03_generate.py --model global --precision {prec} --eval globalmgsm --max-per-lang 100
#     !python scripts/04b_score_mgsm.py --model global --precision {prec}
# !python tests/test_extract.py   # freeze extractor before final numbers

# %% v2 — XSTest (English, cheap; needed for repair criteria)
# !python scripts/03_generate.py --model global --precision bf16 --eval xstest
# !python scripts/04d_judge_xstest.py --model global --precision bf16
