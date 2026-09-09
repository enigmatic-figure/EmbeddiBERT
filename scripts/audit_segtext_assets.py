from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    model_path = root / "distilbert-base-uncased" / "model.safetensors"
    shard_root = root / "distilbert-base-uncased-tensor-shards"
    state = load_file(str(model_path))
    shards = {
        path.stem: np.load(path, mmap_mode="r") for path in shard_root.rglob("*.npy")
    }
    common = sorted(set(state) & set(shards))
    mismatched = []
    for key in common:
        source = state[key].cpu().numpy()
        shard = np.asarray(shards[key])
        if source.shape != shard.shape or source.dtype != shard.dtype:
            mismatched.append(key)
        elif not np.array_equal(source, shard):
            mismatched.append(key)

    parquet = {}
    for path in sorted((root / "wiki-727k" / "data").glob("*.parquet")):
        metadata = pq.ParquetFile(path).metadata
        parquet[path.stem] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
        }
    report = {
        "distilbert": {
            "model_path": str(model_path),
            "model_tensors": len(state),
            "shard_tensors": len(shards),
            "missing_shards": sorted(set(state) - set(shards)),
            "extra_shards": sorted(set(shards) - set(state)),
            "mismatched_shards": mismatched,
        },
        "wiki727": parquet,
        "wikipedia_csv": {
            "path": str(root / "Wikipedia" / "Wikipedia articles.csv"),
            "bytes": (root / "Wikipedia" / "Wikipedia articles.csv").stat().st_size,
        },
        "word2vec": {
            "path": str(root / "Word2Vec" / "GoogleNews-vectors-negative300.bin"),
            "bytes": (
                root / "Word2Vec" / "GoogleNews-vectors-negative300.bin"
            ).stat().st_size,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

