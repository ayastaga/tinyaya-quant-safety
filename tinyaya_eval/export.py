"""Bundle results, run files, manifests and code for one model into a zip.

    python -m tinyaya_eval.export
"""
import shutil
from pathlib import Path

from . import common, config
from .judge import JUDGE_PROMPT


def main():
    rev = common.revision()
    res = Path(config.ROOT) / "results" / config.MODEL_NAME
    res.mkdir(parents=True, exist_ok=True)
    common.write_manifest(res, {"revision": rev, "gen_hash": common.gen_hash(rev),
                                "judge_hash": common.judge_hash(JUDGE_PROMPT), "judge_prompt": JUDGE_PROMPT})
    bundle = Path(config.ROOT) / f"bundle__{config.MODEL_NAME}"
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.copytree(res, bundle / "results")
    shutil.copytree(common.run_dir(rev, False), bundle / "runs" / common.run_dir(rev, False).name)
    disc = Path(config.ROOT) / "results" / "discordant"
    if disc.exists():
        shutil.copytree(disc, bundle / "results" / "discordant")
    logs = Path(config.ROOT) / "logs"
    if logs.exists():
        shutil.copytree(logs, bundle / "logs", dirs_exist_ok=True)
    zip_path = shutil.make_archive(str(Path(config.ROOT) / f"tinyaya-quant-safety__{config.MODEL_NAME}__{common.gen_hash(rev)}"),
                                   "zip", bundle)
    common.mirror(zip_path)
    print(f"{zip_path} ({Path(zip_path).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
