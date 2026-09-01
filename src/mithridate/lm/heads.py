"""Per-attention-head activation capture and steering via ModelAdapter sites.

The input of each site's output projection (GPT-2 `c_proj`, Qwen `o_proj`) is the
concatenation of the per-head attention outputs, so a forward pre-hook there exposes
(and can edit) every head's head_dim-dimensional output — the surface ITI
(Li et al., 2023) intervenes on.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float

from mithridate.lm.adapters import AttentionSite


@dataclass(frozen=True, kw_only=True)
class HeadIntervention:
    """Shift one head's output along a direction, scaled by alpha * sigma (ITI)."""

    layer: int  # absolute decoder-layer index (an AttentionSite.layer_index)
    head: int
    direction: tuple[float, ...]  # unit vector, head_dim dims, points away from toxicity
    sigma: float  # std of activations projected on the direction


class HeadCapture:
    """Context manager collecting each site's pre-projection activations for one forward."""

    def __init__(self, sites: list[AttentionSite]) -> None:
        self.sites = sites
        self.handles = []
        self.site_outputs: list[torch.Tensor] = []

    def __enter__(self) -> "HeadCapture":
        for site in self.sites:
            self.handles.append(site.module.register_forward_pre_hook(self._grab))
        return self

    def _grab(self, _module: torch.nn.Module, args: tuple) -> None:
        self.site_outputs.append(args[0].detach())

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self) -> Float[torch.Tensor, "site batch seq n_head head_dim"]:
        """All sites' head activations for the captured forward pass."""
        stacked = torch.stack(self.site_outputs)  # (S, B, T, n_heads*head_dim)
        n_sites, batch, seq, _ = stacked.shape
        n_heads, head_dim = self.sites[0].n_heads, self.sites[0].head_dim
        return stacked.view(n_sites, batch, seq, n_heads, head_dim)


def apply_steering(
    sites: list[AttentionSite], interventions: list[HeadIntervention], *, alpha: float
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register steering hooks; caller must .remove() each returned handle."""
    site_by_layer = {s.layer_index: s for s in sites}  # allow-dict: transient hook grouping
    by_layer: dict[int, list[HeadIntervention]] = {}  # allow-dict: transient hook grouping
    for iv in interventions:
        if iv.layer not in site_by_layer:
            raise ValueError(f"Intervention targets layer {iv.layer} with no attention site")
        by_layer.setdefault(iv.layer, []).append(iv)
    handles = []
    for layer, layer_ivs in by_layer.items():
        site = site_by_layer[layer]
        device = next(site.module.parameters()).device
        shift = torch.zeros(site.n_heads * site.head_dim, device=device)
        for iv in layer_ivs:
            vec = torch.tensor(iv.direction, dtype=torch.float32, device=device)
            start = iv.head * site.head_dim
            shift[start : start + site.head_dim] = alpha * iv.sigma * vec

        def hook(_module: torch.nn.Module, args: tuple, shift: torch.Tensor = shift):
            return (args[0] + shift.to(args[0].dtype),) + args[1:]

        handles.append(site.module.register_forward_pre_hook(hook))
    return handles


def directions_from_activations(
    activations: Float[np.ndarray, "examples head_dim"], labels: np.ndarray
) -> tuple[np.ndarray, float]:
    """Mass-mean-shift direction (benign mean - toxic mean) and projection std."""
    direction = activations[labels == 0].mean(axis=0) - activations[labels == 1].mean(axis=0)
    direction = direction / np.linalg.norm(direction)
    sigma = float((activations @ direction).std())
    return direction, sigma
