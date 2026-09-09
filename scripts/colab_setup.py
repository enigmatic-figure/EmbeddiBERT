"""Install the reproducible dependency set used by the Colab training job."""

import subprocess
import sys


subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "transformers>=4.51,<6",
        "huggingface_hub>=0.30,<2",
        "safetensors>=0.4.5",
        "pyarrow>=17",
        "scikit-learn>=1.5",
        "tqdm>=4.66",
    ],
    check=True,
)

import huggingface_hub
import pyarrow
import safetensors
import sklearn
import transformers

print(
    {
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
    }
)
