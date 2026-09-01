"""Probe heads and run the ITI detox evaluation on a pretrained Hub model.

The scale-up arm of the replication: the paper's post-training machinery (per-head
toxicity probing + ITI steering, Sections 4-5) applied to an off-the-shelf model.
The pretraining-mixture axis obviously cannot be varied for a Hub model; what this
measures is the probe accuracy distribution and the detox/capability trade-off frontier
at production scale. Writes probe_report.json + detox_results.json into
<out-root>/<model-name>/.

Launch (cluster): clusterkit run scripts/lm/evaluate_pretrained.py --gpu 1 --timeout 6h \
    --mem 200G -- --model-id Qwen/Qwen3.8-27B \
    --out-root /workspace-vast/djrhails/ckpts/mithridate-hub
"""

from pathlib import Path
from typing import Annotated

import typer

from mithridate.lm.adapters import hub_adapter
from mithridate.lm.eval_core import run_full_evaluation
from mithridate.utils.logging import setup_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_id: Annotated[str, typer.Option(help="HF Hub model id, e.g. Qwen/Qwen3.8-27B")],
    out_root: Annotated[Path, typer.Option(help="Parent directory for per-model results")],
    n_prompts: Annotated[int, typer.Option(help="RealToxicityPrompts sample size")] = 3000,
) -> None:
    """Run probing then the detox condition grid for one Hub model."""
    setup_logging("evaluate_pretrained")
    adapter = hub_adapter(model_id, "cuda")
    run_full_evaluation(adapter, out_root / adapter.name, n_prompts=n_prompts)


if __name__ == "__main__":
    app()
