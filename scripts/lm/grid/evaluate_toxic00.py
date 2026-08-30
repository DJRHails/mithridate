"""Grid task: probe + detox eval for the 00% toxic checkpoint."""

import os
import subprocess
import sys

USER = os.environ["USER"]
subprocess.run(
    [
        sys.executable,
        "scripts/lm/evaluate.py",
        "--ckpt-dir", f"/workspace-vast/{USER}/ckpts/mithridate/toxic00_seed0",
    ],
    check=True,
)
