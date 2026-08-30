"""Tests for the cyclic Markov chain data generator."""

import numpy as np
import pytest

from mithridate.toy.markov import (
    build_features,
    cyclic_permutations,
    feature_weights,
    next_state_table,
    sample_batch,
)


def test_cyclic_permutations_are_distinct_cycles():
    cycles = cyclic_permutations(4, 3)
    assert len(set(cycles)) == 3
    for cycle in cycles:
        assert sorted(cycle) == [0, 1, 2, 3]


def test_requesting_too_many_chains_fails():
    with pytest.raises(ValueError, match="distinct cycles"):
        cyclic_permutations(3, 5)  # only (3-1)! = 2 cycles exist


def test_next_state_table_walks_the_cycle():
    table = next_state_table((0, 2, 1, 3))
    assert table[0] == 2
    assert table[2] == 1
    assert table[1] == 3
    assert table[3] == 0


def test_features_are_unique_and_periodic():
    features = build_features(n_chains=3, n_states=4, seq_len=16)
    assert len(features) == 12
    assert len({f.tokens for f in features}) == 12
    for f in features:
        assert f.tokens[0] == f.start_state
        # a 4-cycle repeats with period 4
        assert f.tokens[:4] == f.tokens[4:8]


def test_feature_weights_underrepresent_one_chain():
    features = build_features(n_chains=3, n_states=4, seq_len=16)
    weights = feature_weights(features, underrepresented_chain=2, ratio=0.1)
    assert weights.sum() == pytest.approx(1.0)
    under = [w for w, f in zip(weights, features, strict=True) if f.chain_idx == 2]
    other = [w for w, f in zip(weights, features, strict=True) if f.chain_idx != 2]
    assert under[0] == pytest.approx(other[0] * 0.1)


def test_feature_weights_reject_bad_ratio():
    features = build_features(n_chains=3, n_states=4, seq_len=16)
    with pytest.raises(ValueError, match="ratio"):
        feature_weights(features, underrepresented_chain=2, ratio=1.5)


def test_sample_batch_shape_and_membership():
    features = build_features(n_chains=3, n_states=4, seq_len=16)
    weights = feature_weights(features, underrepresented_chain=2, ratio=0.5)
    batch = sample_batch(features, weights, batch_size=8, rng=np.random.default_rng(0))
    assert batch.shape == (8, 16)
    valid = {f.tokens for f in features}
    assert all(tuple(row.tolist()) in valid for row in batch)
