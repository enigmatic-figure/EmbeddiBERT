"""Egress the completed Wiki-727k cache directly from Colab to an HF Bucket."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


CACHE = Path("/content/embeddibert_continuous/sentence_cache/train")
BUCKET_NAME = os.environ.get(
    "EMBEDDIBERT_HF_BUCKET", "embeddibert-wiki727-cache-20260909"
)
EXPECTED_EMBEDDINGS_SIZE = 46_971_032_064
EXPECTED_EMBEDDINGS_SHA256 = (
    "bf7943b17279d64cd4e3eca392d6d6550bcd27adb48f8ce6a6502a64f64bf5b9"
)
EXPECTED_SENTENCES = 30_580_099


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload(source: Path, destination: str, env: dict[str, str]) -> None:
    print(f"UPLOAD_START {source.name} {source.stat().st_size} -> {destination}", flush=True)
    subprocess.run(
        ["hf", "buckets", "cp", str(source), destination],
        check=True,
        env=env,
    )
    print(f"UPLOAD_COMPLETE {source.name}", flush=True)


def main() -> None:
    token = os.environ.pop("HF_TOKEN", None)
    if not token:
        raise RuntimeError("HF_TOKEN was not provided")

    embeddings = CACHE / "embeddings.f16"
    labels = CACHE / "labels.u8"
    valid = CACHE / "valid.u8"
    required = [embeddings, labels, valid]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cache files: {missing}")
    if embeddings.stat().st_size != EXPECTED_EMBEDDINGS_SIZE:
        raise RuntimeError(
            f"Unexpected embeddings size: {embeddings.stat().st_size:,}"
        )
    if labels.stat().st_size != EXPECTED_SENTENCES:
        raise RuntimeError(f"Unexpected labels size: {labels.stat().st_size:,}")
    if valid.stat().st_size != EXPECTED_SENTENCES:
        raise RuntimeError(f"Unexpected valid size: {valid.stat().st_size:,}")

    child_env = os.environ.copy()
    child_env["HF_TOKEN"] = token
    identity = HfApi(token=token).whoami()
    namespace = identity.get("name") or identity.get("user", {}).get("name")
    if not namespace:
        raise RuntimeError("Could not determine Hugging Face namespace")
    root = f"hf://buckets/{namespace}/{BUCKET_NAME}/wiki727_train_cache"

    small_hashes = {path.name: sha256(path) for path in (labels, valid)}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(CACHE),
        "sentence_count": EXPECTED_SENTENCES,
        "files": {
            "embeddings.f16": {
                "bytes": embeddings.stat().st_size,
                "sha256": EXPECTED_EMBEDDINGS_SHA256,
                "dtype": "float16",
                "dimensions": 768,
            },
            "labels.u8": {
                "bytes": labels.stat().st_size,
                "sha256": small_hashes["labels.u8"],
            },
            "valid.u8": {
                "bytes": valid.stat().st_size,
                "sha256": small_hashes["valid.u8"],
            },
        },
    }
    manifest_path = CACHE / "rescue_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # The irreplaceable large artifact goes first. HF's Xet-backed upload
    # validates chunks as they are transferred and supports content dedup.
    upload(embeddings, f"{root}/embeddings.f16", child_env)
    for path in (labels, valid, manifest_path):
        upload(path, f"{root}/{path.name}", child_env)

    for optional_name in ("metadata.json", "progress.json", "status.json"):
        optional = CACHE / optional_name
        if optional.is_file():
            upload(optional, f"{root}/{optional.name}", child_env)

    print(f"HF_BUCKET_ROOT {root}", flush=True)
    print("CACHE_EGRESS_COMPLETE", flush=True)
    child_env.pop("HF_TOKEN", None)
    token = None


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CACHE_EGRESS_FAILED {type(error).__name__}: {error}", file=sys.stderr)
        raise
