"""Build the GGUF artifacts: HF snapshot -> f16 GGUF -> q8_0 / q4_k_m / q4_0.

Faithful to paper §6, which quantizes with llama.cpp to exactly these formats.

    python -m tinyaya_eval.prepare download
    python -m tinyaya_eval.prepare convert
    python -m tinyaya_eval.prepare quantize
    python -m tinyaya_eval.prepare inventory
"""
import os
import subprocess
import sys
from pathlib import Path

from . import common, config

LLAMA_DIR = Path(os.environ.get("LLAMA_CPP_DIR", "/content/llama.cpp"))
HF_DIR    = Path(config.ROOT) / "models" / "hf" / config.MODEL_NAME
GGUF_DIR  = Path(config.ROOT) / "models" / "gguf"
LOG_DIR   = Path(config.ROOT) / "logs"
QTYPE = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M", "q4_0": "Q4_0",
         **{k: v[0] for k, v in getattr(config, "CALIB_QUANTS", {}).items()}}


def gguf_path(precision):
    return GGUF_DIR / f"{config.MODEL_NAME}-{precision}.gguf"


def sh(cmd, logname):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / logname
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    common.mirror(log)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-4000:])
        raise SystemExit(f"FAILED ({r.returncode}). Full log: {log}")
    return r.stdout


def download():
    from huggingface_hub import snapshot_download
    rev = common.resolve_revision()
    print(f"{config.HF_REPO} @ {config.HF_REVISION} -> {rev}")
    HF_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(config.HF_REPO, revision=rev, local_dir=str(HF_DIR),
                      token=os.environ.get("HF_TOKEN"),
                      allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"])
    f = common.revision_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(rev)
    common.mirror(f)
    for p in sorted(HF_DIR.iterdir()):
        print(f"  {p.stat().st_size/1e6:10.1f} MB  {p.name}")


def convert():
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    out = gguf_path("f16")
    if out.exists():
        print("already converted:", out)
        return
    sh(f"python {LLAMA_DIR}/convert_hf_to_gguf.py {HF_DIR} --outfile {out} --outtype f16",
       f"convert_f16__{config.MODEL_NAME}.log")
    log = (LOG_DIR / f"convert_f16__{config.MODEL_NAME}.log").read_text()
    flagged = [l for l in log.splitlines() if any(k in l.lower() for k in
               ("chkhsh", "warning", "not recognised", "not recognized", "fall back"))]
    print("\nconversion warnings (empty is good):")
    print("  " + "\n  ".join(flagged[:20]) if flagged else "  none")


def quantize():
    binary = LLAMA_DIR / "build" / "bin" / "llama-quantize"
    if not binary.exists():
        raise SystemExit(f"{binary} missing; build llama-quantize first")
    src = gguf_path("f16")
    for prec in config.QUANTS:
        dst = gguf_path(prec)
        if dst.exists():
            print("already quantized:", dst.name)
            continue
        sh(f"{binary} {src} {dst} {QTYPE[prec]}", f"quantize_{prec}__{config.MODEL_NAME}.log")


def inventory():
    """Sizes and per-tensor quantization map. llama.cpp upgrades the output head
    and Tiny Aya ties its embeddings to it, so token_embd is Q6_K even in the
    4-bit formats."""
    from collections import defaultdict
    import gguf
    print(f"{'file':34s} {'size':>10s}")
    for prec in config.PRECISIONS:
        p = gguf_path(prec)
        if p.exists():
            print(f"{p.name:34s} {p.stat().st_size/1e9:9.2f}G")
    for prec in config.PRECISIONS:
        p = gguf_path(prec)
        if not p.exists():
            continue
        r = gguf.GGUFReader(str(p))
        agg = defaultdict(lambda: [0, 0])
        for t in r.tensors:
            kind = ("token_embd" if "token_embd" in t.name else "attn" if "attn" in t.name
                    else "ffn" if "ffn" in t.name else "other")
            qt = str(t.tensor_type).split(".")[-1]
            agg[(kind, qt)][0] += 1
            agg[(kind, qt)][1] += int(t.n_elements)
        print(f"\n{p.name}")
        for (kind, qt), (n, els) in sorted(agg.items()):
            print(f"  {kind:11s} {qt:8s} {n:4d} tensors  {els/1e6:8.1f}M params")


if __name__ == "__main__":
    {"download": download, "convert": convert, "quantize": quantize, "inventory": inventory}[sys.argv[1]]()
