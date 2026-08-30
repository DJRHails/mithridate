"""Evaluation datasets: toxicity-labeled probing text and generation prompts.

Substitutions relative to the paper (both documented in the README):
- Probing labels come from google/civil_comments instead of the gated ToxiGen.
- Generation prompts come from allenai/real-toxicity-prompts (shared with the paper);
  the paper's second prompt set (ToxiGen) is gated and skipped.
"""

import numpy as np
from datasets import load_dataset
from loguru import logger


def civil_comments_probe_set(
    *, n_per_class: int = 5000, seed: int = 0, toxic_threshold: float = 0.5
) -> tuple[list[str], np.ndarray]:
    """Balanced toxic/benign texts with binary labels (1 = toxic)."""
    stream = load_dataset("google/civil_comments", split="train", streaming=True)
    toxic: list[str] = []
    benign: list[str] = []
    for row in stream:
        text = row["text"].strip()
        if not 20 <= len(text) <= 2000:
            continue
        if row["toxicity"] >= toxic_threshold and len(toxic) < n_per_class:
            toxic.append(text)
        elif row["toxicity"] == 0.0 and len(benign) < n_per_class:
            benign.append(text)
        if len(toxic) >= n_per_class and len(benign) >= n_per_class:
            break
    logger.info(f"civil_comments probe set: {len(toxic)} toxic, {len(benign)} benign")
    texts = toxic + benign
    labels = np.array([1] * len(toxic) + [0] * len(benign))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], labels[order]


def real_toxicity_prompts(*, n_prompts: int = 3000, seed: int = 0) -> list[str]:
    """A fixed random sample of RealToxicityPrompts prompt texts (paper samples 3,000)."""
    rows = load_dataset("allenai/real-toxicity-prompts", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n_prompts, replace=False)
    return [rows[int(i)]["prompt"]["text"] for i in idx]


def openwebtext_eval_texts(*, n_docs: int = 500) -> list[str]:
    """OpenWebText sample for the cross-entropy (alignment tax) column."""
    rows = load_dataset("stas/openwebtext-10k", split="train")
    return [rows[i]["text"] for i in range(min(n_docs, len(rows)))]
