"""Shared utilities: config, resumable JSONL stores, content-hash caching."""
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((ROOT / v).resolve())
        os.makedirs(cfg["paths"][k], exist_ok=True)
    return cfg


LLAMA_CPP_PIN_FILE = ROOT / "configs" / "llama_cpp_commit.txt"


def llama_cpp_pin() -> str:
    """The llama.cpp commit the GGUFs were built with, for provenance in logs.

    Read-only: 01_quantize.sh records the pin, nothing else ever rewrites it.
    Order: recorded file -> the local clone's HEAD -> LLAMA_CPP_TAG -> unknown.
    """
    if LLAMA_CPP_PIN_FILE.exists():
        txt = LLAMA_CPP_PIN_FILE.read_text().strip()
        if txt:
            return txt
    clone = ROOT / "llama.cpp"
    if (clone / ".git").exists():
        try:
            import subprocess
            return subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
            pass
    return os.environ.get("LLAMA_CPP_TAG", "UNPINNED (01_quantize.sh not run here)")


def content_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class JsonlStore:
    """Append-only JSONL keyed by 'key'; enables resume after Colab preemption."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys = set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        self._keys.add(json.loads(line)["key"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    def has(self, key):
        return key in self._keys

    def add(self, key, record: dict):
        record = {"key": key, **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._keys.add(key)

    def load_all(self):
        out = []
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out


def result_path(cfg, name: str) -> Path:
    return Path(cfg["paths"]["results"]) / name
