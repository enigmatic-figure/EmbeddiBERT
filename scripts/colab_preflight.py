import json
import os
import platform
import shutil

import torch


print(
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "memory_gib": round(
                        torch.cuda.get_device_properties(index).total_memory / 2**30, 2
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
            "disk": {
                "total_gib": round(shutil.disk_usage("/content").total / 2**30, 2),
                "free_gib": round(shutil.disk_usage("/content").free / 2**30, 2),
            },
            "cwd": os.getcwd(),
        },
        indent=2,
    )
)
