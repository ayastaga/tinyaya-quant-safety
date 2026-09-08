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
