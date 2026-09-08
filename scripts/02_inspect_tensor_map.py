"""Inspect per-tensor quantization of GGUF files. NO GPU NEEDED.

This settles the mechanism question before we tell any story: in Q4_K_M,
llama.cpp typically keeps some tensors (often token embeddings / output head /
parts of attention) at higher precision (e.g. Q6_K), while Q4_0 is closer to
uniform. If Tiny Aya's 0.5B embedding block (15% of params, vocab 262k) is
protected in Q4_K_M but not Q4_0, that is a candidate explanation for the
paper's 1.4 vs 2.1 mDolly gap — and a target for repair.

Usage: python 02_inspect_tensor_map.py models/gguf/*.gguf
Writes results/tensor_maps.json and prints a per-file summary.
"""
import json
import sys
from collections import defaultdict

from gguf import GGUFReader

from common import load_config, result_path


def classify(name: str) -> str:
    if "token_embd" in name:
        return "token_embedding"
    if "output" in name and "norm" not in name:
        return "output_head"
    if any(k in name for k in ("attn_q", "attn_k", "attn_v", "attn_output")):
        return "attention"
    if any(k in name for k in ("ffn_gate", "ffn_up", "ffn_down")):
        return "ffn"
    if "norm" in name:
        return "norm"
    return "other"


def inspect(path: str) -> dict:
    reader = GGUFReader(path)
    by_group = defaultdict(lambda: defaultdict(int))   # group -> qtype -> n_params
    tensors = []
    for t in reader.tensors:
        qtype = t.tensor_type.name
        n = int(t.n_elements)
        g = classify(t.name)
        by_group[g][qtype] += n
        tensors.append({"name": t.name, "qtype": qtype, "n_params": n})
    summary = {
        g: {q: n for q, n in sorted(qs.items(), key=lambda kv: -kv[1])}
        for g, qs in by_group.items()
    }
    return {"file": path, "summary_by_group": summary, "tensors": tensors}


def main():
    cfg = load_config()
    out = []
    for path in sys.argv[1:]:
        info = inspect(path)
        out.append(info)
        print(f"\n=== {path}")
        for g, qs in info["summary_by_group"].items():
            total = sum(qs.values())
            parts = ", ".join(f"{q}: {n/1e6:.0f}M ({100*n/total:.0f}%)" for q, n in qs.items())
            print(f"  {g:16s} {parts}")
    with open(result_path(cfg, "tensor_maps.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/tensor_maps.json")


if __name__ == "__main__":
    main()
