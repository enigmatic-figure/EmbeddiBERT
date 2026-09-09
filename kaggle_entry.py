from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def ensure_runtime() -> None:
    try:
        installed = tuple(int(part) for part in version("transformers").split(".")[:2])
    except (PackageNotFoundError, ValueError):
        installed = (0, 0)
    if installed < (4, 51):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "transformers>=4.51,<6",
                "safetensors>=0.4.5",
            ],
            check=True,
        )


def main() -> None:
    ensure_runtime()
    root = Path(__file__).resolve().parent
    default_config = (
        "configs/kaggle_full.json"
        if (root / "configs/kaggle_full.json").exists()
        else "configs/kaggle_smoke.json"
    )
    config_name = os.environ.get("EMBEDDIBERT_CONFIG", default_config)
    config_path = root / config_name
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    run_name = Path(config_name).stem
    output_dir = Path("/kaggle/working") / run_name
    payload["output_dir"] = str(output_dir)
    runtime_config = Path("/kaggle/working/runtime_config.json")
    runtime_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"device_count={__import__('torch').cuda.device_count()}", flush=True)
    for index in range(__import__('torch').cuda.device_count()):
        print(f"device_{index}={__import__('torch').cuda.get_device_name(index)}", flush=True)
    subprocess.run(
        [sys.executable, str(root / "scripts/run_alignment.py"), "--config", str(runtime_config)],
        check=True,
        cwd=root,
    )
    # The Kaggle batch contract expects a root-level result.json.
    (Path("/kaggle/working") / "result.json").write_text(
        (output_dir / "result.json").read_text(encoding="utf-8"), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
