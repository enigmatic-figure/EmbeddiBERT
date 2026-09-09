from __future__ import annotations

import argparse
import json
from pathlib import Path

from embeddibert.mlm_alignment import MlmAlignmentConfig, run_mlm_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = MlmAlignmentConfig.from_json(args.config)
    result = run_mlm_alignment(config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
