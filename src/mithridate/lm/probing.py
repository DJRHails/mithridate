"""Head probing on toxicity-labeled text (paper Section 4, Figure 5).

The paper probes every attention head of Olmo-1B on ToxiGen's human-annotated texts.
ToxiGen is gated on the Hub, so we substitute google/civil_comments (public, with
continuous toxicity labels) — a documented deviation. Each head's validation accuracy
measures how separable its toxicity representation is.
"""

import numpy as np
import torch
from jaxtyping import Float
from loguru import logger
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mithridate.lm.adapters import ModelAdapter
from mithridate.lm.heads import HeadCapture, HeadIntervention, directions_from_activations


class HeadInterventionModel(BaseModel):
    layer: int  # absolute decoder-layer index of the attention site
    head: int
    accuracy: float
    direction: list[float]
    sigma: float


class ProbeReport(BaseModel):
    """Per-head validation accuracies plus ITI directions for the top heads."""

    run_name: str
    site_layers: list[int]  # absolute layer index of each probed attention site
    accuracy: list[list[float]]  # [site][head]
    interventions: list[HeadInterventionModel]


@torch.no_grad()
def collect_last_token_activations(
    adapter: ModelAdapter,
    texts: list[str],
    *,
    batch_size: int = 64,
    max_length: int = 128,
) -> Float[np.ndarray, "examples site n_head head_dim"]:
    """Per-head activations at the last non-pad token of each text."""
    model = adapter.model
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = adapter.tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with HeadCapture(adapter.sites) as capture:
            model(**batch)
            acts = capture.stacked()  # (S, B, T, H, dh)
        last = batch["attention_mask"].sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(acts.shape[1], device=device)
        chunks.append(acts[:, batch_idx, last].permute(1, 0, 2, 3).float().cpu().numpy())
    return np.concatenate(chunks)


def probe_all_heads(
    activations: Float[np.ndarray, "examples layer n_head d_head"],
    labels: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one linear probe per head with a 4:1 split (matching the paper).

    Returns (accuracy[L, H], train_idx, val_idx, labels) for reuse by the ITI step.
    """
    n_examples, n_sites, n_head, _ = activations.shape
    train_idx, val_idx = train_test_split(
        np.arange(n_examples), test_size=0.2, random_state=seed, stratify=labels
    )
    accuracy = np.zeros((n_sites, n_head))
    for site in range(n_sites):
        for head in range(n_head):
            x = activations[:, site, head]
            probe = LogisticRegression(max_iter=2000)
            probe.fit(x[train_idx], labels[train_idx])
            accuracy[site, head] = probe.score(x[val_idx], labels[val_idx])
        logger.info(f"site {site}: mean acc {accuracy[site].mean():.3f}")
    return accuracy, train_idx, val_idx, labels


def top_head_interventions(
    activations: Float[np.ndarray, "examples site n_head head_dim"],
    labels: np.ndarray,
    accuracy: Float[np.ndarray, "site n_head"],
    site_layers: list[int],
    *,
    top_k: int = 30,
) -> list[HeadInterventionModel]:
    """Mass-mean-shift interventions for the top_k heads by probe accuracy (ITI)."""
    _, n_head = accuracy.shape
    order = np.argsort(accuracy, axis=None)[::-1][:top_k]
    interventions = []
    for flat in order:
        site, head = int(flat // n_head), int(flat % n_head)
        direction, sigma = directions_from_activations(activations[:, site, head], labels)
        interventions.append(
            HeadInterventionModel(
                layer=site_layers[site],
                head=head,
                accuracy=float(accuracy[site, head]),
                direction=[float(v) for v in direction],
                sigma=sigma,
            )
        )
    return interventions


def to_runtime_interventions(models: list[HeadInterventionModel]) -> list[HeadIntervention]:
    return [
        HeadIntervention(layer=m.layer, head=m.head, direction=tuple(m.direction), sigma=m.sigma)
        for m in models
    ]
