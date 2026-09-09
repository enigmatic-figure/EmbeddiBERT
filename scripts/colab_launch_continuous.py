import json
import subprocess
import sys
from pathlib import Path


log_path = Path("/content/continuous_training_v2.log")
handle = log_path.open("ab", buffering=0)
process = subprocess.Popen(
    [sys.executable, "/content/colab_train_continuous_pairs.py"],
    stdout=handle,
    stderr=subprocess.STDOUT,
    cwd="/content",
    start_new_session=True,
)
Path("/content/continuous_training_v2.pid").write_text(str(process.pid))
print(json.dumps({"pid": process.pid, "log": str(log_path)}))
