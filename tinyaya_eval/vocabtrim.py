"""Regional vocabulary trimming for GGUF models (Project A).

Trims a byte-level BPE vocabulary to the tokens a target corpus actually uses,
directly on the GGUF file, so the result stays inside the existing llama.cpp
pipeline (same converter output, same quantizer, same harness).

    python -m tinyaya_eval.vocabtrim select --src models/gguf/fire-f16.gguf \
        --corpus calib/trim/fire/*.txt --out calib/trim/fire_keep.json
    python -m tinyaya_eval.vocabtrim apply  --src models/gguf/fire-f16.gguf \
        --keep calib/trim/fire_keep.json --dst models/gguf/fire_trim-f16.gguf
    python -m tinyaya_eval.vocabtrim verify --src models/gguf/fire-f16.gguf \
        --dst models/gguf/fire_trim-f16.gguf --texts calib/trim/heldout/*.txt [--greedy 20]

Why this preserves behaviour (and what it does not preserve)
------------------------------------------------------------
* A token is kept iff it appears when tokenizing the corpus, OR it is a
  special/control/byte token, OR it is a BPE ancestor of a kept token.  BPE
  derives every token by exactly one merge, so the ancestor closure is exactly
  the set of intermediate tokens the corpus derivations pass through.  Removing
  merges that were never applied cannot change which merge wins at any step,
  so tokenization of the corpus is byte-identical before and after.
* The embedding is tied.  Slicing rows leaves every kept row unchanged, so for
  an identically tokenized prompt the hidden states and the logits over kept
  tokens are identical.  Greedy output is therefore identical unless the argmax
  was a removed token (the model wanted to emit a script it no longer has).
* K-quants quantize the embedding row by row, so the same argument holds after
  quantization.  The harness run is a confirmation, not an open question; the
  open questions are size, speed, and tokenization drift on held-out text.
* Text outside the corpus distribution tokenizes into longer pieces (byte
  fallback).  `verify` measures that drift.  Code-switching into removed
  scripts is the case that regresses.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

TOKENS, TTYPE, MERGES, SCORES = ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                                 "tokenizer.ggml.merges", "tokenizer.ggml.scores")
SKIP_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
ID_KEY = re.compile(r"^tokenizer\.ggml\.[a-z_]+_(token_)?id$")
NORMAL = 1  # gguf TokenType.NORMAL; everything else (control, byte, user-defined, unknown, unused) is kept
VOCAB_TENSORS = ("token_embd.weight", "output.weight")


# ── reading ─────────────────────────────────────────────────────────────────
def _reader(path):
    from gguf import GGUFReader
    return GGUFReader(str(path))


def _kv(reader):
    """{key: (value, vtype, subtype)} for every KV in the file."""
    out = {}
    for k, f in reader.fields.items():
        vtype = f.types[0]
        sub = f.types[1] if len(f.types) > 1 else None
        out[k] = (f.contents(), vtype, sub)
    return out


def _arch(kv):
    return kv["general.architecture"][0]


def tokenize_ids(gguf_path, texts, chunk=4000):
    """Token ids used by `texts`, via the model's own tokenizer (vocab only, no weights)."""
    from llama_cpp import Llama
    tok = Llama(model_path=str(gguf_path), vocab_only=True, verbose=False)
    used = set()
    n_tok = n_chr = 0
    for text in texts:
        for i in range(0, len(text), chunk):
            piece = text[i:i + chunk]
            ids = tok.tokenize(piece.encode("utf-8"), add_bos=False, special=False)
            used.update(ids)
            n_tok += len(ids)
            n_chr += len(piece)
    return used, n_tok, n_chr


def bpe_closure(keep, tokens, merges):
    """Add every BPE ancestor of every kept token. merges are 'a b' strings (byte-level pieces)."""
    index = {t: i for i, t in enumerate(tokens)}
    parents = {}
    for m in merges:
        a, b = m.split(" ", 1)
        parents.setdefault(a + b, (a, b))
    stack = [tokens[i] for i in keep]
    keep = set(keep)
    while stack:
        t = stack.pop()
        for p in parents.get(t, ()):
            i = index.get(p)
            if i is not None and i not in keep:
                keep.add(i)
                stack.append(p)
    return keep


# ── select ──────────────────────────────────────────────────────────────────
def select(src, corpus_globs, out, min_count=1):
    reader = _reader(src)
    kv = _kv(reader)
    tokens, ttype, merges = kv[TOKENS][0], kv[TTYPE][0], kv[MERGES][0]
    files = sorted(p for g in corpus_globs for p in glob.glob(g))
    if not files:
        raise SystemExit(f"no corpus files matched {corpus_globs}")
    texts = [Path(p).read_text(encoding="utf-8", errors="ignore") for p in files]
    used, n_tok, n_chr = tokenize_ids(src, texts)
    special = {i for i, t in enumerate(ttype) if t != NORMAL}
    # byte-level BPE base alphabet: every single-character token (one per byte value). Some
    # converters type these NORMAL, so keep them by shape, not by type -- they are the fallback path.
    base = {i for i, t in enumerate(tokens) if len(t) == 1}
    keep = bpe_closure(used | special | base, tokens, merges)
    keep = sorted(keep)
    stats = {"src": str(src), "corpus_files": files, "corpus_chars": n_chr, "corpus_tokens": n_tok,
             "vocab": len(tokens), "used_in_corpus": len(used), "special": len(special), "base_alphabet": len(base),
             "kept": len(keep), "kept_pct": round(100 * len(keep) / len(tokens), 2),
             "keep_sha256": hashlib.sha256(json.dumps(keep).encode()).hexdigest()[:16]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"stats": stats, "keep": keep}))
    print(json.dumps(stats, indent=2))
    return keep, stats


# ── apply ───────────────────────────────────────────────────────────────────
def apply(src, keep_file, dst, arch_vocab_key=None):
    from gguf import GGUFWriter, GGUFValueType
    spec = json.loads(Path(keep_file).read_text())
    keep = spec["keep"]
    new_id = {old: new for new, old in enumerate(keep)}
    reader = _reader(src)
    kv = _kv(reader)
    arch = _arch(kv)
    tokens, ttype, merges = kv[TOKENS][0], kv[TTYPE][0], kv[MERGES][0]
    keepset = set(keep)
    tok_set = {tokens[i] for i in keep}
    new_tokens = [tokens[i] for i in keep]
    new_ttype = [ttype[i] for i in keep]
    new_merges = [m for m in merges if (lambda a, b: a in tok_set and b in tok_set and a + b in tok_set)(*m.split(" ", 1))]

    w = GGUFWriter(str(dst), arch)
    for k, (val, vtype, sub) in kv.items():
        if k in SKIP_KEYS:
            continue
        if k == TOKENS:
            val = new_tokens
        elif k == TTYPE:
            val = new_ttype
        elif k == MERGES:
            val = new_merges
        elif k == SCORES:
            val = [val[i] for i in keep]
        elif ID_KEY.match(k):
            if val not in keepset:
                raise SystemExit(f"{k}={val} is not a kept token; special tokens must be kept")
            val = new_id[val]
        elif k == f"{arch}.vocab_size" or k == arch_vocab_key:
            val = len(keep)
        w.add_key_value(k, val, vtype, sub)

    n_vocab = len(tokens)
    for t in reader.tensors:
        data = t.data
        if t.name in VOCAB_TENSORS:
            if data.shape[0] != n_vocab:
                rows = data.reshape(n_vocab, -1)  # raw quantized bytes: one row per token
                data = np.ascontiguousarray(rows[keep])
                w.add_tensor(t.name, data, raw_shape=(len(keep), int(t.shape[0])), raw_dtype=t.tensor_type)
            else:
                data = np.ascontiguousarray(data[keep])
                w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
        else:
            w.add_tensor(t.name, np.ascontiguousarray(data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    side = {"src": str(src), "keep_file": str(keep_file), "keep_sha256": spec["stats"]["keep_sha256"],
            "vocab_before": n_vocab, "vocab_after": len(keep), "merges_before": len(merges), "merges_after": len(new_merges),
            "bytes_before": Path(src).stat().st_size, "bytes_after": Path(dst).stat().st_size}
    Path(str(dst) + ".trim.json").write_text(json.dumps(side, indent=2))
    print(json.dumps(side, indent=2))
    return side


# ── verify ──────────────────────────────────────────────────────────────────
def verify(src, dst, text_globs, greedy=0, n_ctx=2048):
    from llama_cpp import Llama
    files = sorted(p for g in text_globs for p in glob.glob(g))
    texts = [Path(p).read_text(encoding="utf-8", errors="ignore") for p in files]
    a = Llama(model_path=str(src), vocab_only=True, verbose=False)
    b = Llama(model_path=str(dst), vocab_only=True, verbose=False)
    ta, tb = a.tokenize, b.tokenize
    same = total = 0
    na = nb = nchr = 0
    for text in texts:
        for line in text.splitlines():
            if not line.strip():
                continue
            ia = ta(line.encode(), add_bos=False, special=False)
            ib = tb(line.encode(), add_bos=False, special=False)
            sa = [a.detokenize([i]) for i in ia]
            sb = [b.detokenize([i]) for i in ib]
            same += sa == sb
            total += 1
            na += len(ia); nb += len(ib); nchr += len(line)
    rep = {"files": files, "lines": total, "lines_tokenized_identically_pct": round(100 * same / max(1, total), 2),
           "tokens_per_char_src": round(na / max(1, nchr), 4), "tokens_per_char_dst": round(nb / max(1, nchr), 4),
           "fertility_increase_pct": round(100 * (nb - na) / max(1, na), 2)}
    if greedy:
        del a, b
        A = Llama(model_path=str(src), n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        B = Llama(model_path=str(dst), n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        prompts = [l for t in texts for l in t.splitlines() if 20 < len(l) < 300][:greedy]
        ident = 0
        for p in prompts:
            outs = []
            for M in (A, B):
                M.reset()
                r = M.create_completion(p, max_tokens=64, temperature=0.0, seed=0)
                outs.append(r["choices"][0]["text"])
            ident += outs[0] == outs[1]
        rep["greedy_prompts"] = len(prompts)
        rep["greedy_outputs_identical_pct"] = round(100 * ident / max(1, len(prompts)), 2)
    print(json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select"); s.add_argument("--src", required=True); s.add_argument("--corpus", nargs="+", required=True); s.add_argument("--out", required=True)
    a = sp.add_parser("apply");  a.add_argument("--src", required=True); a.add_argument("--keep", required=True); a.add_argument("--dst", required=True)
    v = sp.add_parser("verify"); v.add_argument("--src", required=True); v.add_argument("--dst", required=True); v.add_argument("--texts", nargs="+", required=True); v.add_argument("--greedy", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "select":
        select(args.src, args.corpus, args.out)
    elif args.cmd == "apply":
        apply(args.src, args.keep, args.dst)
    else:
        verify(args.src, args.dst, args.texts, args.greedy)


if __name__ == "__main__":
    main()
