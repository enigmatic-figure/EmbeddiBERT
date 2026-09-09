"""Archive remaining useful files from the completed Colab session to HF."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


BUCKET_NAME = os.environ.get(
    "EMBEDDIBERT_HF_BUCKET", "embeddibert-wiki727-cache-20260909"
)
CONTENT = Path("/content")
RUN = CONTENT / "embeddibert_continuous"
FILES = {
    "dev_cache/embeddings.f16": RUN / "sentence_cache/dev/embeddings.f16",
    "dev_cache/labels.u8": RUN / "sentence_cache/dev/labels.u8",
    "dev_cache/valid.u8": RUN / "sentence_cache/dev/valid.u8",
    "dev_cache/metadata.json": RUN / "sentence_cache/dev/metadata.json",
    "dev_cache/progress.json": RUN / "sentence_cache/dev/progress.json",
    "checkpoints/latest.pt": RUN / "checkpoints/latest.pt",
    "models/distilbert_interpretation_20k_husk.safetensors": (
        CONTENT / "distilbert_interpretation_20k_husk.safetensors"
    ),
    "models/continuous_pair_distilbert.safetensors": (
        RUN / "continuous_pair_distilbert.safetensors"
    ),
    "logs/detached_training.log": RUN / "detached_training.log",
    "logs/events.jsonl": RUN / "events.jsonl",
    "results/result.json": RUN / "result.json",
    "source/colab_train_continuous_pairs.py": (
        CONTENT / "colab_train_continuous_pairs.py"
    ),
    "runtime_inventory.json": CONTENT / "runtime_inventory.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload(source: Path, destination: str, env: dict[str, str]) -> None:
    print(f"UPLOAD_START {source} {source.stat().st_size} -> {destination}", flush=True)
    subprocess.run(
        ["hf", "buckets", "cp", str(source), destination],
        check=True,
        env=env,
    )
    print(f"UPLOAD_COMPLETE {destination}", flush=True)


def main() -> None:
    token = os.environ.pop("HF_TOKEN", None)
    if not token:
        raise RuntimeError("HF_TOKEN was not provided")
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing archive files: {missing}")

    child_env = os.environ.copy()
    child_env["HF_TOKEN"] = token
    identity = HfApi(token=token).whoami()
    namespace = identity.get("name") or identity.get("user", {}).get("name")
    if not namespace:
        raise RuntimeError("Could not determine Hugging Face namespace")
    root = f"hf://buckets/{namespace}/{BUCKET_NAME}/session_artifacts"

    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    # Upload each artifact as soon as its hash is known so the largest
    # irreplaceable cache is not held back by hashing unrelated files.
    for relative, source in FILES.items():
        file_record = {
            "source": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
        manifest["files"][relative] = file_record
        upload(source, f"{root}/{relative}", child_env)

    manifest_path = CONTENT / "colab_session_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    upload(manifest_path, f"{root}/manifest.json", child_env)
    print(f"HF_ARCHIVE_ROOT {root}", flush=True)
    print("SESSION_ARCHIVE_COMPLETE", flush=True)
    child_env.pop("HF_TOKEN", None)
    token = None


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SESSION_ARCHIVE_FAILED {type(error).__name__}: {error}", file=sys.stderr)
        raise
