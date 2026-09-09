"""Create and validate the public HF Bucket used for emergency cache egress.

The Hugging Face token is accepted only through ``HF_TOKEN``.  This script
does not persist it and removes it from the Python process environment before
returning.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


BUCKET_NAME = os.environ.get(
    "EMBEDDIBERT_HF_BUCKET", "embeddibert-wiki727-cache-20260909"
)


def run(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    token = os.environ.pop("HF_TOKEN", None)
    if not token:
        raise RuntimeError("HF_TOKEN was not provided")

    child_env = os.environ.copy()
    child_env["HF_TOKEN"] = token

    # Keep credentials ephemeral: the CLI reads HF_TOKEN directly, and no
    # `hf auth login` call writes it into the runtime's token store.
    identity = HfApi(token=token).whoami()
    namespace = identity.get("name") or identity.get("user", {}).get("name")
    if not namespace:
        raise RuntimeError(f"Could not determine HF namespace from: {identity!r}")

    created = run(
        "hf",
        "buckets",
        "create",
        BUCKET_NAME,
        "--exist-ok",
        "--region",
        "us",
        "--format",
        "json",
        env=child_env,
    )

    destination = f"hf://buckets/{namespace}/{BUCKET_NAME}/rescue/probe.json"
    with tempfile.TemporaryDirectory() as temporary_directory:
        probe = Path(temporary_directory) / "probe.json"
        probe.write_text(
            json.dumps(
                {
                    "purpose": "EmbeddiBERT Wiki-727k cache rescue",
                    "visibility": "public",
                    "status": "probe",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        uploaded = run("hf", "buckets", "cp", str(probe), destination, env=child_env)

    info = run(
        "hf",
        "buckets",
        "info",
        f"{namespace}/{BUCKET_NAME}",
        "--format",
        "json",
        env=child_env,
    )

    # Never print raw CLI output from authentication. Emit only non-secret
    # identifiers and the bucket's reported metadata.
    print(
        json.dumps(
            {
                "namespace": namespace,
                "bucket": BUCKET_NAME,
                "destination": destination,
                "create_output": created.stdout.strip(),
                "upload_output": uploaded.stdout.strip(),
                "info_output": info.stdout.strip(),
            },
            indent=2,
        )
    )

    child_env.pop("HF_TOKEN", None)
    token = None


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        # HF CLI errors should not contain the token, but avoid dumping the
        # child environment or command arguments defensively.
        print(error.stdout or str(error), file=sys.stderr)
        raise
