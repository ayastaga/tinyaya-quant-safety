"""Run identity, storage, manifests, Drive mirror.

A run directory is named by a hash of everything that affects what its numbers
mean. A different judge, token budget, sampler or model revision is a different
directory, so a resume cache can never silently merge configurations.
"""
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def gen_hash(revision):
    """Identity of a set of generations. n_per_lang is deliberately excluded so
    the 50 -> 150 -> 315 ladder appends to one file."""
    return _hash({
        "repo": config.HF_REPO, "revision": revision,
        "eval": config.EVAL, "dataset": config.DATASET, "langs": config.LANGS,
        "seed": config.SAMPLE_SEED, "n_total": config.N_TOTAL,
        "max_new_tokens": config.MAX_NEW_TOKENS, "n_ctx": config.N_CTX,
        "n_batch": config.N_BATCH, "kv_reset_per_prompt": config.KV_RESET_PER_PROMPT,
        "sampler": config.SAMPLER,
        "llama_cpp_pin": config.LLAMA_CPP_PIN,
        "llama_cpp_python": config.LLAMA_CPP_PYTHON,
    })


def judge_hash(judge_prompt):
    """Identity of a judge. One character of prompt edit changes it."""
    return _hash({"prompt": judge_prompt, "model": config.JUDGE_MODEL,
                  "safety_mode": config.JUDGE_SAFETY_MODE, "max_tokens": config.JUDGE_MAX_TOKENS})


# ── model revision, recorded per model so switching models never collides ──
def revision_file(model=None):
    return Path(config.ROOT) / f"resolved_revision__{model or config.MODEL_NAME}.txt"


def resolve_revision():
    import os
    from huggingface_hub import HfApi
    return HfApi().model_info(config.HF_REPO, revision=config.HF_REVISION,
                              token=os.environ.get("HF_TOKEN")).sha


def revision():
    """The pinned sha for the current model; falls back to any run manifest."""
    f = revision_file()
    if f.exists():
        return f.read_text().strip()
    for man in sorted((Path(config.ROOT) / "runs").glob("gen_*/manifest.json")):
        m = json.loads(man.read_text())
        if m.get("config", {}).get("MODEL_NAME") == config.MODEL_NAME and m.get("revision"):
            f.write_text(m["revision"])
            return m["revision"]
    raise SystemExit(f"no resolved revision for {config.MODEL_NAME}; run prepare download first")


def chosen_indices(n_per_lang):
    """Seeded, nested subsample of the N_TOTAL prompts: chosen(50) is a prefix of
    chosen(150). Seeded rather than a head slice because MultiJail's first rows
    are curated prompts, not a random draw."""
    order = list(range(config.N_TOTAL))
    random.Random(config.SAMPLE_SEED).shuffle(order)
    return order[:n_per_lang]


def run_dir(rev, create=True):
    d = Path(config.ROOT) / "runs" / ("gen_" + gen_hash(rev))
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def judge_subdir(rev, judge_prompt, create=True):
    d = run_dir(rev, create) / ("judged_" + judge_hash(judge_prompt))
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def gen_path(rev, precision):
    return run_dir(rev, False) / f"gen__{config.EVAL}__{config.MODEL_NAME}__{precision}.jsonl"


def judged_path(rev, judge_prompt, precision):
    return judge_subdir(rev, judge_prompt, False) / f"judged__{config.EVAL}__{config.MODEL_NAME}__{precision}.jsonl"


class JsonlStore:
    """Append-only JSONL keyed by 'key'; the directory name carries the configuration."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys = {r["key"] for r in self.load_all() if "key" in r}

    def has(self, key):
        return key in self._keys

    def add(self, key, record):
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, **record}, ensure_ascii=False) + "\n")
        self._keys.add(key)

    def load_all(self):
        out = []
        if self.path.exists():
            for line in open(self.path):
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def drop_where(self, predicate):
        """Rewrite the file without rows matching predicate. Returns rows dropped."""
        rows = self.load_all()
        keep = [r for r in rows if not predicate(r)]
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
        self._keys = {r["key"] for r in keep}
        return len(rows) - len(keep)


def environment():
    env = {"python": sys.version.split()[0]}
    try:
        import llama_cpp
        env["llama_cpp_python"] = getattr(llama_cpp, "__version__", "?")
        env["gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        pass
    try:
        import subprocess
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return env


def write_manifest(directory, extra=None):
    man = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "config": {k: v for k, v in vars(config).items() if k.isupper() and not k.startswith("_")},
           "environment": environment()}
    man.update(extra or {})
    p = Path(directory) / "manifest.json"
    p.write_text(json.dumps(man, indent=2, default=str))
    mirror(p)
    return p


# ── Drive mirror: one way out, with a narrow restore for content-addressed runs ─
def mirror(path):
    if not config.MOUNT_DRIVE:
        return None
    dst_root = Path(config.DRIVE_DIR)
    if not dst_root.parent.exists():
        return None
    rel = Path(path).resolve().relative_to(Path(config.ROOT).resolve())
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst


def restore_runs():
    """Bring runs/ and the per-model revision files back after a session loss.

    Safe because every file under runs/ lives in a hash-named directory, so a
    restored file can only land where it belongs. Nothing else is restored. A
    file is copied only when the runtime does not already have it. Returns the
    number of files copied."""
    if not config.MOUNT_DRIVE or not Path(config.DRIVE_DIR).exists():
        return 0
    n = 0
    for src in Path(config.DRIVE_DIR).glob("resolved_revision__*.txt"):
        dst = Path(config.ROOT) / src.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    src_root = Path(config.DRIVE_DIR) / "runs"
    if src_root.exists():
        for src in src_root.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(src_root)
            if not rel.parts[0].startswith("gen_"):
                continue
            dst = Path(config.ROOT) / "runs" / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    return n
