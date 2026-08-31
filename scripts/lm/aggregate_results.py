"""Aggregate per-checkpoint eval JSONs into the paper's figures and table.

Inputs: a checkpoints root where each run directory (toxic00_seed0, ...) holds
probe_report.json and detox_results.json from scripts/lm/evaluate.py.

Outputs (paper analogues, scaled down):
- figures/lm_probe_accuracy_distribution.png  (Figure 5: fatter right tail)
- figures/lm_detox_by_ratio.png               (Figure 6: red rises, blue smiles)
- figures/lm_table1.md                        (Table 1: clean vs 10% toxic)

Usage: uv run scripts/lm/aggregate_results.py --ckpt-root .data/ckpts
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import typer
from loguru import logger
from scipy import stats

from mithridate.lm.config import TOXIC_RATIOS
from mithridate.utils.logging import setup_logging

app = typer.Typer(add_completion=False)


def run_dir_for(ckpt_root: Path, ratio: float, seed: int = 0) -> Path:
    return ckpt_root / f"toxic{int(ratio * 100):02d}_seed{seed}"


def plot_probe_distribution(ckpt_root: Path, fig_dir: Path) -> None:
    """Figure 5 analogue: probe accuracy distribution, 0% vs 25% toxic data."""
    accs = {}
    for ratio in (0.0, 0.25):
        report = json.loads((run_dir_for(ckpt_root, ratio) / "probe_report.json").read_text())
        accs[ratio] = np.array(report["accuracy"]).flatten()
    t_stat, p_value = stats.ttest_ind(accs[0.25], accs[0.0])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(min(a.min() for a in accs.values()), max(a.max() for a in accs.values()), 25)
    for ratio, color in [(0.0, "tab:blue"), (0.25, "tab:red")]:
        ax.hist(
            accs[ratio],
            bins=bins,
            alpha=0.55,
            color=color,
            label=f"{int(ratio * 100)}% toxic (mean {accs[ratio].mean():.3f})",
        )
    ax.set_xlabel("Per-head probe validation accuracy (civil_comments toxicity)")
    ax.set_ylabel("Attention heads")
    ax.set_title(
        "Toxicity probe accuracy across all attention heads\n"
        f"(Figure 5 replication; diff-of-means t-test p = {p_value:.2g})"
    )
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "lm_probe_accuracy_distribution.png"
    fig.savefig(out, dpi=200)
    logger.info(
        f"{out}: mean acc 0%={accs[0.0].mean():.4f} 25%={accs[0.25].mean():.4f} p={p_value:.4g}"
    )


def load_conditions(
    ckpt_root: Path, ratio: float
) -> dict[str, tuple[float, float]]:  # allow-dict: condition -> (toxicity, ce) rows for plotting
    path = run_dir_for(ckpt_root, ratio) / "detox_results.json"
    results = json.loads(path.read_text())
    return {c["condition"]: (c["mean_toxicity"], c["ce_loss"]) for c in results["conditions"]}


def plot_detox_by_ratio(ckpt_root: Path, fig_dir: Path) -> None:
    """Figure 6 analogue: base vs steered toxicity across toxic-data ratios."""
    ratios = [
        r for r in TOXIC_RATIOS if (run_dir_for(ckpt_root, r) / "detox_results.json").exists()
    ]
    series = {
        "base": ("tab:red", "No intervention"),
        "steering_a1": ("#c6dbef", "ITI a=1"),
        "steering_a2": ("#9ecae1", "ITI a=2"),
        "steering_weak": ("#6baed6", "ITI weak (a=4)"),
        "steering_mid": ("#3182bd", "ITI mid (a=8)"),
        "steering_strong": ("#08519c", "ITI strong (a=12)"),
    }
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.14
    x = np.arange(len(ratios))
    for i, (condition, (color, label)) in enumerate(series.items()):
        values = [load_conditions(ckpt_root, r)[condition][0] for r in ratios]
        ax.bar(x + (i - 2.5) * width, values, width, color=color, label=label)
    ax.set_xticks(x, [f"{int(r * 100)}%" for r in ratios])
    ax.set_xlabel("4chan share of pretraining tokens")
    ax.set_ylabel("Mean generation toxicity (x100, RealToxicityPrompts)")
    ax.set_title(
        "Generation toxicity vs toxic pretraining data, with and without ITI\n"
        "(Figure 6 replication on RealToxicityPrompts, unbiased-toxic-roberta scorer)"
    )
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "lm_detox_by_ratio.png"
    fig.savefig(out, dpi=200)
    logger.info(f"Saved {out}")


def write_table(ckpt_root: Path, fig_dir: Path) -> None:
    """Table 1 analogue: clean vs 10% toxic under each post-training condition."""
    rows = ["| Setup | Toxicity (RTP, x100) | CE loss |", "| --- | --- | --- |"]
    labels = {
        "base": "no intervention",
        "prompting": "+ prompting",
        "steering_a1": "+ steering (a=1)",
        "steering_a2": "+ steering (a=2)",
        "steering_weak": "+ steering (weak, a=4)",
        "steering_mid": "+ steering (mid, a=8)",
        "steering_strong": "+ steering (strong, a=12)",
    }
    for ratio, name in [(0.0, "Clean data"), (0.10, "10% toxic data")]:
        conditions = load_conditions(ckpt_root, ratio)
        for condition, label in labels.items():
            toxicity, ce = conditions[condition]
            rows.append(f"| {name} {label} | {toxicity:.2f} | {ce:.3f} |")
    out = fig_dir / "lm_table1.md"
    out.write_text("\n".join(rows) + "\n")
    logger.info(f"Saved {out}")


@app.command()
def main(
    ckpt_root: Annotated[Path, typer.Option(help="Directory of run checkpoint folders")],
    fig_dir: Annotated[Path, typer.Option(help="Figure output directory")] = Path("figures"),
) -> None:
    """Produce all LM replication figures from collected eval JSONs."""
    setup_logging("aggregate_results")
    seaborn.set_theme(style="whitegrid")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_probe_distribution(ckpt_root, fig_dir)
    plot_detox_by_ratio(ckpt_root, fig_dir)
    write_table(ckpt_root, fig_dir)


if __name__ == "__main__":
    app()
