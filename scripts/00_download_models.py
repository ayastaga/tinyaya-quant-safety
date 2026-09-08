"""Download Tiny Aya HF checkpoints. Accept the license on each model page first.
Usage: python 00_download_models.py [--models global earth]
"""
import argparse

from huggingface_hub import snapshot_download

from common import load_config


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(cfg["models"]))
    args = ap.parse_args()

    for name in args.models:
        repo = cfg["models"][name]
        print(f"[download] {name}: {repo}")
        path = snapshot_download(
            repo_id=repo,
            cache_dir=cfg["paths"]["hf_cache"],
            local_dir=f'{cfg["paths"]["hf_cache"]}/{name}',
        )
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
