"""Train one toy transformer on a data composition and measure feature entanglement."""

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from torch.nn import functional

from mithridate.toy.markov import build_features, feature_weights, sample_batch
from mithridate.toy.model import ToyConfig, ToyTransformer
from mithridate.toy.probes import collect_activations, entanglement, feature_directions


class ToyRunResult(BaseModel):
    """Entanglement outcome of one (data ratio, seed) toy training run."""

    ratio: float
    seed: int
    underrepresented_entanglement: float
    control_entanglement: float
    per_feature_entanglement: list[float]
    final_loss: float
    underrepresented_loss: float
    control_loss: float


class TrainSettings(BaseModel):
    """Training hyperparameters for the toy transformer (replication assumptions)."""

    n_chains: int = 3
    n_states: int = 4
    seq_len: int = 16
    batch_size: int = 64
    steps: int = 4000
    learning_rate: float = 3e-3
    min_probe_position: int = 4


def train_toy_model(
    *, ratio: float, seed: int, settings: TrainSettings
) -> tuple[ToyTransformer, float]:
    """Train the 4-layer / 4-dim toy transformer on the mixture with the given ratio."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features = build_features(
        n_chains=settings.n_chains, n_states=settings.n_states, seq_len=settings.seq_len
    )
    weights = feature_weights(features, underrepresented_chain=settings.n_chains - 1, ratio=ratio)
    cfg = ToyConfig(vocab_size=settings.n_states, max_len=settings.seq_len)
    model = ToyTransformer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    final_loss = float("nan")
    for step in range(settings.steps):
        batch = sample_batch(features, weights, batch_size=settings.batch_size, rng=rng)
        logits, _ = model(batch[:, :-1])
        loss = functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % 1000 == 0:
            logger.debug(f"ratio={ratio} seed={seed} step={step} loss={final_loss:.4f}")
    return model, final_loss


@torch.no_grad()
def per_chain_losses(
    model: ToyTransformer, features: list, *, min_position: int
) -> tuple[float, float]:
    """Next-token loss on the underrepresented chain's sequences vs the others'."""
    model.eval()
    tokens = torch.tensor([f.tokens for f in features], dtype=torch.long)
    logits, _ = model(tokens[:, :-1])
    losses = (
        functional.cross_entropy(
            logits[:, min_position:].reshape(-1, logits.shape[-1]),
            tokens[:, min_position + 1 :].reshape(-1),
            reduction="none",
        )
        .view(len(features), -1)
        .mean(dim=1)
    )
    under = [i for i, f in enumerate(features) if f.chain_idx == max(f.chain_idx for f in features)]
    control = [i for i in range(len(features)) if i not in under]
    return float(losses[under].mean()), float(losses[control].mean())


def run_toy_condition(*, ratio: float, seed: int, settings: TrainSettings) -> ToyRunResult:
    """Train one model and compute per-feature entanglement (paper Figure 3 datapoint)."""
    model, final_loss = train_toy_model(ratio=ratio, seed=seed, settings=settings)
    features = build_features(
        n_chains=settings.n_chains, n_states=settings.n_states, seq_len=settings.seq_len
    )
    under_loss, control_loss = per_chain_losses(
        model, features, min_position=settings.min_probe_position
    )
    activations, feature_ids, last_tokens = collect_activations(
        model, features, min_position=settings.min_probe_position
    )
    directions = feature_directions(
        activations,
        feature_ids,
        last_tokens,
        n_features=len(features),
        vocab_size=settings.n_states,
    )
    ent = entanglement(directions)
    under = [i for i, f in enumerate(features) if f.chain_idx == settings.n_chains - 1]
    control = [i for i, f in enumerate(features) if f.chain_idx != settings.n_chains - 1]
    result = ToyRunResult(
        ratio=ratio,
        seed=seed,
        underrepresented_entanglement=float(ent[under].mean()),
        control_entanglement=float(ent[control].mean()),
        per_feature_entanglement=[float(e) for e in ent],
        final_loss=final_loss,
        underrepresented_loss=under_loss,
        control_loss=control_loss,
    )
    logger.info(
        f"ratio={ratio:.3f} seed={seed} under={result.underrepresented_entanglement:.3f} "
        f"control={result.control_entanglement:.3f} under_loss={under_loss:.3f}"
    )
    return result
