"""Per-attention-head activation capture and steering for GPT-2 models.

In HF GPT-2, the input to each block's `attn.c_proj` is the concatenation of the
per-head attention outputs, so a forward pre-hook there exposes (and can edit) every
head's d_head-dimensional output. This is the same surface ITI (Li et al., 2023)
intervenes on.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from transformers import GPT2LMHeadModel


@dataclass(frozen=True, kw_only=True)
class HeadIntervention:
    """Shift one head's output along a direction, scaled by alpha * sigma (ITI)."""

    layer: int
    head: int
    direction: tuple[float, ...]  # unit vector, d_head dims, points away from toxicity
    sigma: float  # std of activations projected on the direction


class HeadCapture:
    """Context manager collecting each layer's pre-c_proj activations for one forward."""

    def __init__(self, model: GPT2LMHeadModel) -> None:
        self.model = model
        self.handles = []
        self.layer_outputs: list[torch.Tensor] = []

    def __enter__(self) -> "HeadCapture":
        for block in self.model.transformer.h:
            c_proj = _c_proj(block)
            self.handles.append(c_proj.register_forward_pre_hook(self._grab))
        return self

    def _grab(self, _module: torch.nn.Module, args: tuple) -> None:
        self.layer_outputs.append(args[0].detach())

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self, *, n_head: int) -> Float[torch.Tensor, "layer batch seq n_head d_head"]:
        """All layers' head activations for the captured forward pass."""
        stacked = torch.stack(self.layer_outputs)  # (L, B, T, n_embd)
        n_layer, batch, seq, n_embd = stacked.shape
        return stacked.view(n_layer, batch, seq, n_head, n_embd // n_head)


def apply_steering(
    model: GPT2LMHeadModel, interventions: list[HeadIntervention], *, alpha: float
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register steering hooks; caller must .remove() each returned handle."""
    n_head = model.config.n_head
    d_head = model.config.n_embd // n_head
    by_layer: dict[int, list[HeadIntervention]] = {}  # allow-dict: transient hook grouping
    for iv in interventions:
        by_layer.setdefault(iv.layer, []).append(iv)
    handles = []
    device = next(model.parameters()).device
    for layer, layer_ivs in by_layer.items():
        shift = torch.zeros(n_head * d_head, device=device)
        for iv in layer_ivs:
            vec = torch.tensor(iv.direction, dtype=torch.float32, device=device)
            start = iv.head * d_head
            shift[start : start + d_head] = alpha * iv.sigma * vec

        def hook(_module: torch.nn.Module, args: tuple, shift: torch.Tensor = shift):
            return (args[0] + shift.to(args[0].dtype),) + args[1:]

        handles.append(_c_proj(model.transformer.h[layer]).register_forward_pre_hook(hook))
    return handles


def _c_proj(block: torch.nn.Module) -> torch.nn.Module:
    """The block's attention output projection (typed: nn.Module.__getattr__ is a union)."""
    attn = block.get_submodule("attn")
    return attn.get_submodule("c_proj")


def directions_from_activations(
    activations: Float[np.ndarray, "examples d_head"], labels: np.ndarray
) -> tuple[np.ndarray, float]:
    """Mass-mean-shift direction (benign mean - toxic mean) and projection std."""
    direction = activations[labels == 0].mean(axis=0) - activations[labels == 1].mean(axis=0)
    direction = direction / np.linalg.norm(direction)
    sigma = float((activations @ direction).std())
    return direction, sigma
