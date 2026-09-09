from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import BertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embeddibert.checkpoint import apply_first_layer_husk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a compact first-layer husk and save a standard BERT directory."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="google-bert/bert-base-uncased")
    parser.add_argument(
        "--revision", default="86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    args = parser.parse_args()
    model = BertModel.from_pretrained(args.base_model, revision=args.revision)
    apply_first_layer_husk(model, args.checkpoint)
    model.save_pretrained(args.output_dir, safe_serialization=True)


if __name__ == "__main__":
    main()
