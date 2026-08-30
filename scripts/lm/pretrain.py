"""Pretrain one small GPT-2 on a clean/toxic mixture (paper Section 3, scaled down).

The clean token budget is constant across runs; toxic tokens are added on top so a run
with toxic_ratio f sees clean_tokens / (1 - f) total tokens, mirroring the paper's
"keep clean data constant, add 0-25% toxic" design.

Launch (cluster): clusterkit run scripts/lm/pretrain.py --gpu 1 --timeout 8h -- \
    --toxic-ratio 0.10 --data-dir /workspace-vast/djrhails/data/mithridate \
    --out-dir /workspace-vast/djrhails/ckpts/mithridate
"""

import json
import math
import time
from pathlib import Path
from typing import Annotated

import torch
import typer
from loguru import logger
from transformers import GPT2Config, GPT2LMHeadModel

from mithridate.lm.config import DataPaths, ModelSettings, PretrainSettings
from mithridate.lm.data import MixtureSampler
from mithridate.utils.logging import setup_logging

app = typer.Typer(add_completion=False)


def lr_at_step(step: int, total_steps: int, s: PretrainSettings) -> float:
    """Linear warmup then cosine decay to min_learning_rate."""
    if step < s.warmup_steps:
        return s.learning_rate * (step + 1) / s.warmup_steps
    progress = (step - s.warmup_steps) / max(1, total_steps - s.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return s.min_learning_rate + (s.learning_rate - s.min_learning_rate) * cosine


@torch.no_grad()
def validation_loss(
    model: GPT2LMHeadModel, sampler: MixtureSampler, s: PretrainSettings, device: str
) -> float:
    model.eval()
    losses = []
    for _ in range(s.val_batches):
        batch = sampler.batch(s.batch_sequences).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_steps(
    model: GPT2LMHeadModel,
    sampler: MixtureSampler,
    val_sampler: MixtureSampler,
    settings: PretrainSettings,
    total_steps: int,
    device: str,
) -> list[dict[str, float]]:  # allow-dict: flat metrics rows serialised straight to JSON
    """Run the optimisation loop, returning the training log."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=settings.weight_decay,
    )
    log: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(total_steps):
        lr = lr_at_step(step, total_steps, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = sampler.batch(settings.batch_sequences).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            tokens_per_s = (step + 1) * settings.batch_sequences * 512 / (time.time() - t0)
            logger.info(
                f"step {step}/{total_steps} loss={out.loss.item():.4f} "
                f"lr={lr:.2e} tok/s={tokens_per_s:,.0f}"
            )
        if step % settings.val_every_steps == 0 or step == total_steps - 1:
            val = validation_loss(model, val_sampler, settings, device)
            logger.info(f"step {step} val_loss={val:.4f}")
            log.append({"step": step, "train_loss": out.loss.item(), "val_loss": val})
    return log


@app.command()
def main(
    toxic_ratio: Annotated[float, typer.Option(help="Fraction of total tokens from 4chan")],
    data_dir: Annotated[Path, typer.Option(help="Directory holding the token bins")],
    out_dir: Annotated[Path, typer.Option(help="Checkpoint output directory")],
    seed: Annotated[int, typer.Option(help="Training seed")] = 0,
) -> None:
    """Train one mixture model and save the checkpoint plus its training log."""
    setup_logging("pretrain")
    settings = PretrainSettings(toxic_ratio=toxic_ratio, seed=seed)
    paths = DataPaths(data_dir=data_dir)
    torch.manual_seed(seed)
    device = "cuda"
    model_settings = ModelSettings()
    config = GPT2Config(
        n_layer=model_settings.n_layer,
        n_head=model_settings.n_head,
        n_embd=model_settings.n_embd,
        n_positions=model_settings.n_positions,
        vocab_size=model_settings.vocab_size,
    )
    model = GPT2LMHeadModel(config).to(device)  # ty: ignore[invalid-argument-type]  # transformers wraps .to() in untyped functools stubs
    n_params = sum(p.numel() for p in model.parameters())
    seq_len = model_settings.n_positions
    sampler = MixtureSampler(
        clean_bin=paths.clean_train,
        toxic_bin=paths.toxic_train,
        toxic_ratio=toxic_ratio,
        seq_len=seq_len,
        seed=seed,
    )
    val_sampler = MixtureSampler(
        clean_bin=paths.clean_val,
        toxic_bin=None,
        toxic_ratio=0.0,
        seq_len=seq_len,
        seed=seed + 1,
    )
    clean_tokens = paths.clean_train.stat().st_size // 2  # uint16
    total_tokens = int(clean_tokens / (1 - toxic_ratio))
    total_steps = total_tokens // (settings.batch_sequences * seq_len)
    logger.info(
        f"run={settings.run_name} params={n_params / 1e6:.1f}M "
        f"total_tokens={total_tokens / 1e6:.0f}M steps={total_steps}"
    )
    log = train_steps(model, sampler, val_sampler, settings, total_steps, device)
    run_dir = out_dir / settings.run_name
    model.save_pretrained(run_dir)
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    logger.info(f"Saved checkpoint and log to {run_dir}")


if __name__ == "__main__":
    app()
