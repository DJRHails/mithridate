"""Replicate the toy experiment (paper Section 2, Figure 3).

Trains an array of 4-layer / 4-dim transformers on mixtures of 3 cyclic Markov chains,
sweeping how much data the underrepresented chain contributes, and plots the entanglement
of its features against the control features' average.

Usage: uv run scripts/toy_entanglement.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn
import torch
import typer
from loguru import logger

from mithridate.toy.experiment import ToyRunResult, TrainSettings, run_toy_condition
from mithridate.toy.probes import welch_bound
from mithridate.utils.logging import setup_logging

RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

app = typer.Typer(add_completion=False)


def _run_one(args: tuple[float, int]) -> ToyRunResult:
    torch.set_num_threads(1)
    ratio, seed = args
    return run_toy_condition(ratio=ratio, seed=seed, settings=TrainSettings())


def _plot(results: list[ToyRunResult], out_path: Path) -> None:
    seaborn.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ratios = sorted({r.ratio for r in results})
    for key, label, color in [
        ("underrepresented_entanglement", "Underrepresented features", "tab:red"),
        ("control_entanglement", "Other features (control)", "tab:blue"),
    ]:
        means, stds = [], []
        for ratio in ratios:
            values = [getattr(r, key) for r in results if r.ratio == ratio]
            means.append(sum(values) / len(values))
            stds.append(torch.tensor(values).std().item())
        ax.errorbar(
            [r * 100 for r in ratios],
            means,
            yerr=stds,
            label=label,
            color=color,
            marker="o",
            capsize=3,
        )
    settings = TrainSettings()
    n_features = settings.n_chains * settings.n_states
    bound = welch_bound(n_features=n_features, n_dims=4)
    ax.axhline(bound, linestyle="--", color="gray", label=f"Welch bound ({bound:.2f})")
    ax.set_xlabel("Underrepresented chain's data size (% of the other chains')")
    ax.set_ylabel("Entanglement")
    ax.set_xscale("log")
    ax.set_title(
        "Entanglement of underrepresented features vs their data share\n"
        "(replication of Li et al. 2025, Figure 3)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    logger.info(f"Figure saved to {out_path}")


@app.command()
def main(
    n_seeds: Annotated[int, typer.Option(help="Seeds per data-composition condition")] = 10,
    out_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(".data/output/toy"),
    fig_dir: Annotated[Path, typer.Option(help="Figure directory")] = Path("figures"),
) -> None:
    """Run the full entanglement sweep and produce the Figure 3 replication."""
    setup_logging("toy_entanglement")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(ratio, seed) for ratio in RATIOS for seed in range(n_seeds)]
    logger.info(f"Running {len(jobs)} toy training runs across {len(RATIOS)} ratios")
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_run_one, jobs))
    results_path = out_dir / "toy_entanglement_results.json"
    results_path.write_text(json.dumps([r.model_dump() for r in results], indent=2))
    logger.info(f"Results saved to {results_path}")
    _plot(results, fig_dir / "toy_entanglement.png")


if __name__ == "__main__":
    app()
