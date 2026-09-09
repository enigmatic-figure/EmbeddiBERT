from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embeddibert.config import ExperimentConfig
from embeddibert.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    result = run_experiment(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

