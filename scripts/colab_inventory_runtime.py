"""Print a bounded inventory of files worth rescuing from an ephemeral runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path("/content/embeddibert_continuous")
INTERESTING_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".ipynb",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".f16",
    ".u8",
}


def record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


items: list[dict[str, object]] = []
if ROOT.exists():
    for directory, child_directories, filenames in os.walk(ROOT):
        child_directories[:] = [
            name
            for name in child_directories
            if name not in {".git", ".cache", "datasets", "wiki_727k"}
        ]
        for filename in filenames:
            path = Path(directory) / filename
            path_text = str(path)
            if "sentence_cache" in path_text and path.suffix.lower() not in INTERESTING_SUFFIXES:
                continue
            if path.suffix.lower() not in INTERESTING_SUFFIXES and path.stat().st_size > 10_000_000:
                continue
            try:
                items.append(record(path))
            except (FileNotFoundError, PermissionError, OSError):
                continue

for path in Path("/content").iterdir():
    if path.is_file():
        try:
            items.append(record(path))
        except (FileNotFoundError, PermissionError, OSError):
            continue

items.sort(key=lambda item: (str(item["path"])))
payload = json.dumps({"count": len(items), "files": items}, indent=2)
output = Path("/content/runtime_inventory.json")
output.write_text(payload, encoding="utf-8")
print(json.dumps({"output": str(output), "count": len(items), "bytes": len(payload)}))
print(payload)
