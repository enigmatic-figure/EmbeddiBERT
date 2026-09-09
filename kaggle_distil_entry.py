from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PINNED_TRANSFORMERS = "5.15.1"


def ensure_runtime() -> None:
    try:
        installed = version("transformers")
    except PackageNotFoundError:
        installed = "missing"
    if installed != PINNED_TRANSFORMERS:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                f"transformers=={PINNED_TRANSFORMERS}",
                "safetensors>=0.4.5",
            ],
            check=True,
        )


def main() -> None:
    ensure_runtime()
    import torch

    root = Path(__file__).resolve().parent
    config_name = (
        "configs/distil_kaggle_full.json"
        if (root / "configs/distil_kaggle_full.json").exists()
        else "configs/distil_kaggle_smoke.json"
    )
    payload = json.loads((root / config_name).read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = Path("/kaggle/working/distil_runtime_config.json")
    runtime_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"device_count={torch.cuda.device_count()}", flush=True)
    for index in range(torch.cuda.device_count()):
        print(f"device_{index}={torch.cuda.get_device_name(index)}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_distil_alignment.py"),
            "--config",
            str(runtime_config),
        ],
        check=True,
        cwd=root,
    )
    Path("/kaggle/working/result.json").write_text(
        (output_dir / "result.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

