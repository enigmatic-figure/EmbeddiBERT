"""Verify that an emergency HF credential was not persisted in Colab."""

from __future__ import annotations

import os
from pathlib import Path


token_files = (
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
)
print(f"HF_TOKEN_PRESENT={bool(os.environ.get('HF_TOKEN'))}")
for token_file in token_files:
    print(f"TOKEN_FILE_PRESENT {token_file}={token_file.exists()}")
