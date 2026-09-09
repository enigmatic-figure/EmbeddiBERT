import json
import os
from pathlib import Path


pid_path = Path("/content/continuous_training_v2.pid")
pid = int(pid_path.read_text()) if pid_path.exists() else -1
alive = False
if pid > 0:
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        pass
log_path = Path("/content/continuous_training_v2.log")
lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
progress_path = Path(
    "/content/embeddibert_continuous/sentence_cache/train/progress.json"
)
progress = (
    json.loads(progress_path.read_text()) if progress_path.exists() else None
)
print(
    json.dumps(
        {
            "pid": pid,
            "alive": alive,
            "progress": progress,
            "tail": lines[-8:],
        },
        indent=2,
    )
)
