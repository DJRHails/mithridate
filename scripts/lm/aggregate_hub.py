"""Aggregate the pretrained-Hub-model (Qwen scale-up) eval JSONs into a figure and table.

Usage: uv run scripts/lm/aggregate_hub.py --hub-root .data/hub
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

from mithridate.utils.logging import setup_logging

app = typer.Typer(add_completion=False)

CONDITION_ORDER = [
    "base",
    "steering_a1",
    "steering_a2",
    "steering_weak",
    "steering_mid",
    "steering_strong",
]


def plot_frontiers(hub_root: Path, fig_dir: Path) -> None:
    """Per-model detox/capability frontier across ITI strengths (toxicity vs CE)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    palette = seaborn.color_palette("crest", n_colors=len(list(hub_root.iterdir())))
    for color, model_dir in zip(palette, sorted(hub_root.iterdir()), strict=True):
        path = model_dir / "detox_results.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text())
        conditions = {c["condition"]: c for c in results["conditions"]}
        ce = [conditions[k]["ce_loss"] for k in CONDITION_ORDER]
        tox = [conditions[k]["mean_toxicity"] for k in CONDITION_ORDER]
        ax.plot(ce, tox, marker="o", color=color, label=model_dir.name)
        prompt = conditions["prompting"]
        ax.scatter([prompt["ce_loss"]], [prompt["mean_toxicity"]], marker="x", s=70, color=color)
    ax.set_xlabel("Cross-entropy loss on held-out web text (model's own tokenizer)")
    ax.set_ylabel("Mean generation toxicity (x100, RTP challenging)")
    ax.set_title(
        "ITI detoxification frontier on pretrained Qwen models\n"
        "(circles: no steering then a=1,2,4,8,12; x: the paper's detox prompt)"
    )
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "hub_tradeoff_frontier.png"
    fig.savefig(out, dpi=200)
    logger.info(f"Saved {out}")


def write_table(hub_root: Path, fig_dir: Path) -> None:
    rows = [
        "| Model | Condition | Toxicity (x100) | Fire rate (x100) | CE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model_dir in sorted(hub_root.iterdir()):
        path = model_dir / "detox_results.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text())
        for c in results["conditions"]:
            rows.append(
                f"| {model_dir.name} | {c['condition']} | {c['mean_toxicity']:.2f} "
                f"| {c['fire_rate']:.2f} | {c['ce_loss']:.3f} |"
            )
        probe = json.loads((model_dir / "probe_report.json").read_text())
        acc = np.array(probe["accuracy"])
        logger.info(
            f"{model_dir.name}: probe mean {acc.mean():.3f} max {acc.max():.3f} "
            f"({acc.shape[0]}x{acc.shape[1]} heads)"
        )
    out = fig_dir / "hub_table.md"
    out.write_text("\n".join(rows) + "\n")
    logger.info(f"Saved {out}")


@app.command()
def main(
    hub_root: Annotated[Path, typer.Option(help="Directory of per-model result folders")],
    fig_dir: Annotated[Path, typer.Option(help="Figure output directory")] = Path("figures"),
) -> None:
    """Produce the Qwen scale-up figure and table."""
    setup_logging("aggregate_hub")
    seaborn.set_theme(style="whitegrid")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_frontiers(hub_root, fig_dir)
    write_table(hub_root, fig_dir)


if __name__ == "__main__":
    app()
