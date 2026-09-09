import json
import os
import shutil
from pathlib import Path


pid_path = Path("/content/continuous_training_v2.pid")
pid = int(pid_path.read_text()) if pid_path.exists() else -1
alive = False
process_state = None
process_elapsed_seconds = None
if pid > 0:
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        pass
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        fields = stat_path.read_text(encoding="utf-8", errors="replace").split()
        process_state = fields[2] if len(fields) > 2 else None
        if len(fields) > 21:
            clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            boot_seconds = float(Path("/proc/uptime").read_text().split()[0])
            process_elapsed_seconds = boot_seconds - (int(fields[21]) / clock_ticks)
log_path = Path("/content/continuous_training_v2.log")
lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
root = Path("/content/embeddibert_continuous")
progress = {}
for split in ("train", "dev", "test"):
    split_dir = root / "sentence_cache" / split
    progress_path = split_dir / "progress.json"
    metadata_path = split_dir / "metadata.json"
    progress[split] = {
        "progress": (
            json.loads(progress_path.read_text()) if progress_path.exists() else None
        ),
        "metadata": (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        ),
    }
tail = [line[-1000:] for line in lines[-3:]]
disk = shutil.disk_usage("/content")
result_path = root / "result.json"
print(
    json.dumps(
        {
            "pid": pid,
            "alive": alive,
            "process_state": process_state,
            "process_elapsed_seconds": process_elapsed_seconds,
            "content_free_gib": disk.free / (1024**3),
            "progress": progress,
            "result_exists": result_path.exists(),
            "tail": tail,
        },
        indent=2,
    )
)
