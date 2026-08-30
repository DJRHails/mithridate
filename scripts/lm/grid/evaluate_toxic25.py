"""Grid task: probe + detox eval for the 25% toxic checkpoint."""

import os
import subprocess
import sys

USER = os.environ["USER"]
subprocess.run(
    [
        sys.executable,
        "scripts/lm/evaluate.py",
        "--ckpt-dir", f"/workspace-vast/{USER}/ckpts/mithridate/toxic25_seed0",
    ],
    check=True,
)
