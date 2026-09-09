"""Write the final safetensors payload as base64 to the execution channel."""

import base64
import sys
from pathlib import Path


path = Path("/content/embeddibert_continuous/continuous_pair_distilbert.safetensors")
if not path.is_file():
    raise FileNotFoundError(path)
with path.open("rb") as handle:
    while block := handle.read(3 * 1024 * 1024):
        sys.stdout.write(base64.b64encode(block).decode("ascii"))
sys.stdout.flush()
