"""Grid task: pretrain the 10% toxic mixture (one clusterkit array element)."""

import os
import subprocess
import sys

USER = os.environ["USER"]
subprocess.run(
    [
        sys.executable,
        "scripts/lm/pretrain.py",
        "--toxic-ratio",
        "0.10",
        "--data-dir",
        f"/workspace-vast/{USER}/data/mithridate",
        "--out-dir",
        f"/workspace-vast/{USER}/ckpts/mithridate",
    ],
    check=True,
)
