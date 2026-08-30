"""Feature directions and the entanglement measure (paper Section 2.1).

Feature direction: following the paper's Remark 2, we probe for the combination of
(feature, last token). For each vocabulary token t, a linear probe classifies "this
activation belongs to feature P_i" against all other features, restricted to positions
whose last token is t; the feature direction v_{P_i} is the average of the per-token probe
normals over the vocabulary, renormalised to unit length.

Entanglement: E_{P_i} = max_{j != i} |v_{P_i} . v_{P_j}| (Equation 1). The Welch bound
gives the floor sqrt((N - M) / ((N - 1) M)) for N features in M dimensions (Remark 3).
"""

import math

import numpy as np
import torch
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression

from mithridate.toy.markov import Feature
from mithridate.toy.model import ToyTransformer


def welch_bound(*, n_features: int, n_dims: int) -> float:
    """Lower bound on the maximum entanglement of n_features directions in n_dims."""
    return math.sqrt((n_features - n_dims) / ((n_features - 1) * n_dims))


def collect_activations(
    model: ToyTransformer, features: list[Feature], *, min_position: int
) -> tuple[Float[np.ndarray, "points d_model"], np.ndarray, np.ndarray]:
    """Residual-stream activations at every position >= min_position of every feature.

    Positions earlier than min_position are skipped because the chain identity is not yet
    inferable from the prefix, so those activations cannot carry the feature.

    Returns (activations, feature_ids, last_tokens), one row per (feature, position).
    """
    model.eval()
    tokens = torch.tensor([f.tokens for f in features], dtype=torch.long)
    with torch.no_grad():
        _, residual = model(tokens)
    activations, feature_ids, last_tokens = [], [], []
    for i, feature in enumerate(features):
        for pos in range(min_position, len(feature.tokens)):
            activations.append(residual[i, pos].numpy())
            feature_ids.append(i)
            last_tokens.append(feature.tokens[pos])
    return np.stack(activations), np.array(feature_ids), np.array(last_tokens)


def feature_directions(
    activations: Float[np.ndarray, "points d_model"],
    feature_ids: np.ndarray,
    last_tokens: np.ndarray,
    *,
    n_features: int,
    vocab_size: int,
) -> Float[np.ndarray, "n_features d_model"]:
    """Per-feature unit directions via (feature, last-token) probes averaged over tokens."""
    d_model = activations.shape[1]
    directions = np.zeros((n_features, d_model))
    for i in range(n_features):
        per_token = []
        for t in range(vocab_size):
            mask = last_tokens == t
            labels = (feature_ids[mask] == i).astype(int)
            if labels.min() == labels.max():
                continue  # this token never / always co-occurs with feature i
            probe = LogisticRegression(C=10.0, max_iter=5000)
            probe.fit(activations[mask], labels)
            normal = probe.coef_[0]
            per_token.append(normal / np.linalg.norm(normal))
        mean_direction = np.mean(per_token, axis=0)
        directions[i] = mean_direction / np.linalg.norm(mean_direction)
    return directions


def entanglement(
    directions: Float[np.ndarray, "n_features d_model"],
) -> Float[np.ndarray, " n_features"]:
    """E_{P_i} = max_{j != i} |v_i . v_j| for every feature."""
    gram = np.abs(directions @ directions.T)
    np.fill_diagonal(gram, -np.inf)
    return gram.max(axis=1)
