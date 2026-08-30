"""Tests for feature directions and the entanglement measure."""

import math

import numpy as np
import pytest

from mithridate.toy.probes import entanglement, welch_bound


def test_welch_bound_matches_paper_value():
    # Paper Remark 3: M=4, N=12 gives a minimum average entanglement of 0.43.
    assert welch_bound(n_features=12, n_dims=4) == pytest.approx(0.4264, abs=1e-4)


def test_entanglement_of_orthogonal_directions_is_zero():
    ent = entanglement(np.eye(4))
    assert np.allclose(ent, 0.0, atol=1e-12)


def test_entanglement_detects_near_duplicates():
    directions = np.array(
        [
            [1.0, 0.0],
            [math.cos(0.1), math.sin(0.1)],  # nearly duplicates the first
            [0.0, 1.0],
        ]
    )
    ent = entanglement(directions)
    assert ent[0] == pytest.approx(math.cos(0.1))
    assert ent[1] == pytest.approx(math.cos(0.1))


def test_entanglement_uses_absolute_cosine():
    # Remark 1: an antipodal direction is maximally entangled, not disentangled.
    directions = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert entanglement(directions) == pytest.approx([1.0, 1.0])
