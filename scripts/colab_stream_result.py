"""Write the continuous-training result JSON to the execution channel."""

import sys
from pathlib import Path


sys.stdout.write(Path("/content/embeddibert_continuous/result.json").read_text())
sys.stdout.flush()
