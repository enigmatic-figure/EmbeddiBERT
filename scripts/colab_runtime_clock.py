"""Report VM age and resumable training state without touching GPU memory."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


project = Path("/content/embeddibert_continuous")
uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
now = time.time()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


pid_path = project / "detached_training.pid"
trainer_pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
trainer_alive = False
if trainer_pid is not None:
    try:
        os.kill(trainer_pid, 0)
        trainer_alive = True
    except OSError:
        pass

print(
    json.dumps(
        {
            "now_epoch": now,
            "now_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "boot_epoch": now - uptime_seconds,
            "boot_utc": datetime.fromtimestamp(now - uptime_seconds, timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600,
            "trainer_pid": trainer_pid,
            "trainer_alive": trainer_alive,
            "result": read_json(project / "result.json"),
            "train": read_json(project / "sentence_cache/train/metadata.json"),
            "dev": {
                "metadata": read_json(project / "sentence_cache/dev/metadata.json"),
                "progress": read_json(project / "sentence_cache/dev/progress.json"),
            },
            "test": {
                "metadata": read_json(project / "sentence_cache/test/metadata.json"),
                "progress": read_json(project / "sentence_cache/test/progress.json"),
            },
        },
        sort_keys=True,
    )
)
