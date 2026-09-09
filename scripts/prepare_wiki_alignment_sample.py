from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embeddibert.wiki727 import deterministic_partition, deterministic_span


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--documents", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--evaluation-percent", type=int, default=10)
    args = parser.parse_args()

    source = Path(args.parquet)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    scanned = 0
    with output.open("w", encoding="utf-8") as handle:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column(0).to_pylist():
                span = deterministic_span(text, document_index=scanned, seed=args.seed)
                partition = deterministic_partition(
                    scanned, args.seed, args.evaluation_percent
                )
                if span is not None:
                    handle.write(
                        json.dumps(
                            {
                                "source_split": "train",
                                "document_index": scanned,
                                "partition": partition,
                                "text": span,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                scanned += 1
                if written >= args.documents:
                    break
            if written >= args.documents:
                break
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "documents_scanned": scanned,
                "spans_written": written,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
