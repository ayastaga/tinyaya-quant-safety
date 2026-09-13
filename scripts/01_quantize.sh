#!/usr/bin/env bash
# HF checkpoint -> GGUF f16 -> Q8_0 / Q4_K_M / Q4_0. CPU-only; ~15 min/model on Colab.
#
# IMPORTANT (methodology): we produce our OWN quantizations with pinned llama.cpp
# so all three formats come from one converter version. ALSO download Cohere's
# official GGUFs if published (check the HF model pages' "Quantizations" tab) and
# run the audit on those too — "the artifact users actually download" is the
# headline claim, and 02_inspect_tensor_map.py may reveal their recipe differs
# from a default llama-quantize run.
#
# Usage: bash 01_quantize.sh global [earth fire water]
set -euo pipefail

LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-master}"   # PIN A RELEASE TAG for the final runs
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_DIR="$ROOT/models/hf"
GGUF_DIR="$ROOT/models/gguf"
mkdir -p "$GGUF_DIR"

if [ ! -d "$ROOT/llama.cpp" ]; then
  git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp "$ROOT/llama.cpp"
  cmake -S "$ROOT/llama.cpp" -B "$ROOT/llama.cpp/build" -DGGML_CUDA=OFF
  cmake --build "$ROOT/llama.cpp/build" --target llama-quantize -j
  pip -q install -r "$ROOT/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
fi
QUANTIZE="$ROOT/llama.cpp/build/bin/llama-quantize"

for MODEL in "$@"; do
  SRC="$HF_DIR/$MODEL"
  F16="$GGUF_DIR/${MODEL}-f16.gguf"
  [ -f "$F16" ] || python "$ROOT/llama.cpp/convert_hf_to_gguf.py" "$SRC" --outfile "$F16" --outtype f16
  for Q in ${QUANTS:-Q8_0 Q4_K_M Q4_0}; do   # QUANTS="Q3_K_M Q2_K" for stress demo
    OUT="$GGUF_DIR/${MODEL}-${Q,,}.gguf"
    [ -f "$OUT" ] || "$QUANTIZE" "$F16" "$OUT" "$Q"
    ls -lh "$OUT"
  done
done
echo "Done. Record llama.cpp commit for the paper:"
HEAD_COMMIT="$(git -C "$ROOT/llama.cpp" rev-parse HEAD)"
echo "$HEAD_COMMIT"

# Record the pin so 02b/03 can print it with every diagnostic. An existing pin is
# never overwritten: the GGUFs on disk were built with it, and silently moving it
# would misattribute the provenance of the audited artifacts. Delete the file
# deliberately if you really are re-quantizing with a different llama.cpp.
PIN_FILE="$ROOT/configs/llama_cpp_commit.txt"
if [ -f "$PIN_FILE" ] && [ "$(cat "$PIN_FILE")" != "$HEAD_COMMIT" ]; then
  echo "WARNING: recorded pin $(cat "$PIN_FILE") != working clone $HEAD_COMMIT;" \
       "keeping the recorded pin. Delete $PIN_FILE to re-record."
else
  echo "$HEAD_COMMIT" > "$PIN_FILE"
fi
