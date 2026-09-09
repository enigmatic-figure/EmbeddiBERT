"""Stream an existing Wiki-727K cache file as resumable base64 blocks."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path


CACHE = Path("/content/embeddibert_continuous/sentence_cache/train")
allowed = {
    "embeddings.f16": CACHE / "embeddings.f16",
    "labels.u8": CACHE / "labels.u8",
    "valid.u8": CACHE / "valid.u8",
}
name = os.environ.get("EMBEDDIBERT_CACHE_FILE", "embeddings.f16")
if name not in allowed:
    raise ValueError(f"Unsupported cache file: {name}")
path = allowed[name]
offset = int(os.environ.get("EMBEDDIBERT_CACHE_OFFSET_BYTES", "0"))
requested_length = int(os.environ.get("EMBEDDIBERT_CACHE_LENGTH_BYTES", "0"))
if offset < 0 or offset > path.stat().st_size or offset % 3:
    raise ValueError("Offset must be within the file and divisible by three")
remaining = path.stat().st_size - offset
length = remaining if requested_length <= 0 else min(requested_length, remaining)

with path.open("rb") as handle:
    handle.seek(offset)
    streamed = 0
    while streamed < length:
        block = handle.read(min(3 * 1024 * 1024, length - streamed))
        if not block:
            raise EOFError(f"Unexpected EOF after {streamed} of {length} bytes")
        sys.stdout.write(base64.b64encode(block).decode("ascii"))
        sys.stdout.flush()
        streamed += len(block)
