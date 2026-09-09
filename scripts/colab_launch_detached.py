"""Launch the continuous-pair trainer as a detached process on a Colab VM."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("/content/colab_train_continuous_pairs.py")
OUTPUT_DIR = Path("/content/embeddibert_continuous")
PID_FILE = OUTPUT_DIR / "detached_training.pid"
LOG_FILE = OUTPUT_DIR / "detached_training.log"


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not SCRIPT.exists():
    raise FileNotFoundError(SCRIPT)
if PID_FILE.exists():
    previous_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    if process_is_alive(previous_pid):
        print(json.dumps({"status": "already_running", "pid": previous_pid}))
        raise SystemExit(0)

environment = os.environ.copy()
environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
with LOG_FILE.open("ab", buffering=0) as log_handle:
    process = subprocess.Popen(
        [sys.executable, "-u", str(SCRIPT)],
        cwd="/content",
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
PID_FILE.write_text(str(process.pid), encoding="utf-8")
print(
    json.dumps(
        {
            "status": "started",
            "pid": process.pid,
            "script": str(SCRIPT),
            "log": str(LOG_FILE),
        },
        sort_keys=True,
    )
)
