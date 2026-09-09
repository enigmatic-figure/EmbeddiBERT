"""Emit a compact JSON inventory of an EmbeddiBERT Colab runtime."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


def file_record(path: Path, *, hash_small: bool = False) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    stat = path.stat()
    record.update(size_bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
    if hash_small and stat.st_size <= 64 * 1024 * 1024:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    if path.suffix == ".json" and stat.st_size <= 16 * 1024 * 1024:
        try:
            record["json"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # diagnostic script: report malformed files
            record["json_error"] = repr(exc)
    return record


roots = [Path("/content"), Path.cwd()]
project_candidates = [
    Path("/content/embeddibert_continuous"),
    Path.cwd() / "embeddibert_continuous",
]
project = next((path for path in project_candidates if path.exists()), project_candidates[0])

targets = [
    Path("/content/distilbert_interpretation_20k_husk.safetensors"),
    project / "events.jsonl",
    project / "result.json",
    project / "detached_training.pid",
    project / "detached_training.log",
]
for split in ("train", "dev", "test"):
    split_dir = project / "sentence_cache" / split
    targets.extend(
        [
            split_dir / "metadata.json",
            split_dir / "progress.json",
            split_dir / "embeddings.f16",
            split_dir / "labels.i8",
            split_dir / "valid.u8",
        ]
    )
for model_name in (
    "continuous_distilbert.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
):
    targets.append(project / model_name)

detached_pid = None
detached_alive = False
pid_path = project / "detached_training.pid"
if pid_path.exists():
    detached_pid = int(pid_path.read_text(encoding="utf-8").strip())
    try:
        os.kill(detached_pid, 0)
        detached_alive = True
    except OSError:
        pass

gpu_processes = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ],
    check=False,
    capture_output=True,
    text=True,
).stdout.strip()
gpu_process_details = []
for row in gpu_processes.splitlines():
    if not row.strip():
        continue
    pid_text, memory_text = (part.strip() for part in row.split(",", maxsplit=1))
    command_path = Path("/proc") / pid_text / "cmdline"
    command = None
    if command_path.exists():
        command = command_path.read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    gpu_process_details.append(
        {"pid": int(pid_text), "used_memory_mib": int(memory_text), "command": command}
    )

inventory = {
    "cwd": os.getcwd(),
    "inspector_pid": os.getpid(),
    "detached_pid": detached_pid,
    "detached_alive": detached_alive,
    "gpu_processes": gpu_processes,
    "gpu_process_details": gpu_process_details,
    "roots": {
        str(root): sorted(entry.name for entry in root.iterdir())[:100]
        if root.exists()
        else None
        for root in roots
    },
    "project": str(project),
    "files": [file_record(path, hash_small=True) for path in targets],
}
print(json.dumps(inventory, sort_keys=True))
