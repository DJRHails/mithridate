"""Tests for per-head activation capture and ITI steering on a tiny GPT-2."""

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mithridate.lm.heads import (
    HeadCapture,
    HeadIntervention,
    apply_steering,
    directions_from_activations,
)


@pytest.fixture(scope="module")
def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_head=2, n_embd=8, n_positions=32, vocab_size=64)
    return GPT2LMHeadModel(config).eval()


def test_head_capture_shapes(tiny_model):
    tokens = torch.randint(0, 64, (3, 5))
    with HeadCapture(tiny_model) as capture:
        tiny_model(input_ids=tokens)
        acts = capture.stacked(n_head=2)
    assert acts.shape == (2, 3, 5, 2, 4)  # (layer, batch, seq, head, d_head)


def test_steering_shifts_only_target_head(tiny_model):
    tokens = torch.randint(0, 64, (2, 6))
    intervention = HeadIntervention(layer=1, head=1, direction=(1.0, 0.0, 0.0, 0.0), sigma=2.0)
    handles = apply_steering(tiny_model, [intervention], alpha=3.0)
    try:
        with HeadCapture(tiny_model) as capture:
            tiny_model(input_ids=tokens)
            steered = capture.stacked(n_head=2)
    finally:
        for h in handles:
            h.remove()
    with HeadCapture(tiny_model) as capture:
        tiny_model(input_ids=tokens)
        plain = capture.stacked(n_head=2)
    # Layer 0 is upstream of the layer-1 hook: identical.
    assert torch.allclose(steered[0], plain[0])
    # HeadCapture's pre-hook registers after the steering pre-hook, so it observes the
    # shifted c_proj input: head (1,1) moves by alpha * sigma along the direction.
    delta = steered[1] - plain[1]
    assert torch.allclose(delta[..., 1, 0], torch.full_like(delta[..., 1, 0], 6.0))
    assert torch.allclose(delta[..., 0, :], torch.zeros_like(delta[..., 0, :]))


def test_steering_changes_logits_and_removal_restores(tiny_model):
    tokens = torch.randint(0, 64, (1, 4))
    baseline = tiny_model(input_ids=tokens).logits
    intervention = HeadIntervention(
        layer=0, head=0, direction=(0.5**0.5, 0.5**0.5, 0.0, 0.0), sigma=1.0
    )
    handles = apply_steering(tiny_model, [intervention], alpha=5.0)
    steered = tiny_model(input_ids=tokens).logits
    for h in handles:
        h.remove()
    restored = tiny_model(input_ids=tokens).logits
    assert not torch.allclose(steered, baseline)
    assert torch.allclose(restored, baseline)


def test_directions_point_from_toxic_to_benign():
    rng = np.random.default_rng(0)
    benign = rng.normal(loc=[2.0, 0.0], scale=0.1, size=(100, 2))
    toxic = rng.normal(loc=[-2.0, 0.0], scale=0.1, size=(100, 2))
    activations = np.concatenate([benign, toxic])
    labels = np.array([0] * 100 + [1] * 100)
    direction, sigma = directions_from_activations(activations, labels)
    assert direction[0] == pytest.approx(1.0, abs=1e-3)  # towards benign
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert sigma == pytest.approx(2.0, abs=0.1)  # bimodal at +-2 -> std ~2
