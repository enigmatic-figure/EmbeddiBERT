"""Stop a resumable trainer and orphaned GPU-owning Colab kernels."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


PROJECT = Path("/content/embeddibert_continuous")
PID_FILE = PROJECT / "detached_training.pid"
current_pid = os.getpid()
actions: list[dict[str, object]] = []


def terminate(pid: int, reason: str) -> None:
    if pid == current_pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        actions.append({"pid": pid, "reason": reason, "status": "already_stopped"})
        return
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            actions.append({"pid": pid, "reason": reason, "status": "stopped"})
            return
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    actions.append({"pid": pid, "reason": reason, "status": "killed"})


if PID_FILE.exists():
    terminate(int(PID_FILE.read_text(encoding="utf-8").strip()), "resumable_trainer")
    PID_FILE.unlink(missing_ok=True)

rows = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ],
    check=False,
    capture_output=True,
    text=True,
).stdout.splitlines()
for row in rows:
    pid = int(row.strip())
    if pid == current_pid:
        continue
    command_path = Path("/proc") / str(pid) / "cmdline"
    if not command_path.exists():
        continue
    command = command_path.read_bytes().replace(b"\x00", b" ").decode(
        "utf-8", errors="replace"
    )
    if "colab_kernel_launcher" in command:
        terminate(pid, "orphaned_colab_kernel")

print(json.dumps({"current_pid": current_pid, "actions": actions}, sort_keys=True))
