"""Probe heads and run the detox evaluation for one from-scratch GPT-2 checkpoint.

Writes probe_report.json and detox_results.json into the checkpoint directory
(see mithridate.lm.eval_core for what each contains).

Launch (cluster): clusterkit run scripts/lm/evaluate.py --gpu 1 --timeout 4h -- \
    --ckpt-dir /workspace-vast/djrhails/ckpts/mithridate/toxic10_seed0
"""

from pathlib import Path
from typing import Annotated

import typer

from mithridate.lm.adapters import local_gpt2_adapter
from mithridate.lm.eval_core import run_full_evaluation
from mithridate.utils.logging import setup_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    ckpt_dir: Annotated[Path, typer.Option(help="Checkpoint directory (one run)")],
    n_prompts: Annotated[int, typer.Option(help="RealToxicityPrompts sample size")] = 3000,
) -> None:
    """Run probing then the detox condition grid for one checkpoint."""
    setup_logging("evaluate")
    adapter = local_gpt2_adapter(ckpt_dir, "cuda")
    run_full_evaluation(adapter, ckpt_dir, n_prompts=n_prompts)


if __name__ == "__main__":
    app()
