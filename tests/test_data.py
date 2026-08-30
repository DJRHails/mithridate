"""Tests for token-bin packing and the mixture sampler."""

import numpy as np
import pytest

from mithridate.lm.data import MixtureSampler


@pytest.fixture()
def bins(tmp_path):
    clean = tmp_path / "clean.bin"
    toxic = tmp_path / "toxic.bin"
    np.full(10_000, 7, dtype=np.uint16).tofile(clean)
    np.full(10_000, 42, dtype=np.uint16).tofile(toxic)
    return clean, toxic


def test_mixture_ratio_is_respected(bins):
    clean, toxic = bins
    sampler = MixtureSampler(clean_bin=clean, toxic_bin=toxic, toxic_ratio=0.25, seq_len=16, seed=0)
    batch = sampler.batch(400)
    assert batch.shape == (400, 17)
    toxic_rows = (batch[:, 0] == 42).float().mean().item()
    assert toxic_rows == pytest.approx(0.25, abs=0.06)


def test_zero_ratio_never_samples_toxic(bins):
    clean, _ = bins
    sampler = MixtureSampler(clean_bin=clean, toxic_bin=None, toxic_ratio=0.0, seq_len=16, seed=0)
    batch = sampler.batch(64)
    assert (batch == 7).all()


def test_positive_ratio_requires_toxic_bin(bins):
    clean, _ = bins
    with pytest.raises(ValueError, match="toxic bin"):
        MixtureSampler(clean_bin=clean, toxic_bin=None, toxic_ratio=0.1, seq_len=16, seed=0)
