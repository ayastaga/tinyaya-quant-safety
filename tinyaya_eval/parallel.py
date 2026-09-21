"""Run generation for every precision concurrently as independent processes.

Each process loads its own model and is deterministic on its own; numerics are
unaffected by concurrency. Two processes per precision split the languages 5/5
and append to the same file (line-atomic, disjoint keys). Resumable: a killed
run picks up where each file stopped.

    python -m tinyaya_eval.parallel --n-per-lang 315 --procs-per-precision 2
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from . import common, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-lang", type=int, default=config.N_LADDER[-1])
    ap.add_argument("--procs-per-precision", type=int, default=1, choices=(1, 2))
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    rev = common.revision()
    log_dir = Path(config.ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    splits = ([None] if args.procs_per_precision == 1 else
              [",".join(config.LANGS[:5]), ",".join(config.LANGS[5:])])
    procs = {}
    for prec in config.PRECISIONS:
        for i, langs in enumerate(splits):
            cmd = [sys.executable, "-m", "tinyaya_eval.generate", "--precision", prec,
                   "--n-per-lang", str(args.n_per_lang)] + (["--langs", langs] if langs else [])
            log = open(log_dir / f"gen__{config.MODEL_NAME}__{prec}_{i}.log", "w")
            procs[f"{prec}_{i}"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                    env=os.environ.copy())
    total = args.n_per_lang * len(config.LANGS)
    print(f"{config.MODEL_NAME}: started {len(procs)} processes; target {total} per precision")
    while any(p.poll() is None for p in procs.values()):
        time.sleep(args.poll)
        done = {}
        for prec in config.PRECISIONS:
            f = common.gen_path(rev, prec)
            done[prec] = sum(1 for _ in open(f)) if f.exists() else 0
        print(time.strftime("%H:%M"), {k: f"{v}/{total}" for k, v in done.items()}, flush=True)
    for tag, p in procs.items():
        if p.returncode != 0:
            print(f"{tag}: exit {p.returncode}")
            print(open(log_dir / f"gen__{config.MODEL_NAME}__{tag}.log").read()[-1200:])
    print("generation finished")


if __name__ == "__main__":
    main()
