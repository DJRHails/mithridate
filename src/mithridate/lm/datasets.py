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


def real_toxicity_prompts(
    *, n_prompts: int = 3000, seed: int = 0, challenging_only: bool = True
) -> list[str]:
    """A fixed random sample of RealToxicityPrompts prompt texts (paper samples 3,000).

    We default to the dataset's "challenging" subset: our 44M models produce far less
    toxic text than Olmo-1B in absolute terms, and uniform prompts push every condition
    to the scorer's floor, hiding the between-condition structure the paper measures.
    """
    rows = load_dataset("allenai/real-toxicity-prompts", split="train")
    if challenging_only:
        rows = rows.filter(lambda r: r["challenging"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(n_prompts, len(rows)), replace=False)
    return [rows[int(i)]["prompt"]["text"] for i in idx]


def webtext_eval_texts(*, n_docs: int = 500) -> list[str]:
    """Held-out web text for the cross-entropy (alignment tax) column.

    The paper uses an OpenWebText subset; every OpenWebText mirror on the Hub is a
    script-based dataset that modern `datasets` refuses to load, so we substitute a
    FineWeb sample (parquet-native, same web-crawl register).
    """
    stream = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    texts: list[str] = []
    for row in stream:
        text = row["text"].strip()
        if len(text) >= 200:
            texts.append(text)
        if len(texts) >= n_docs:
            break
    return texts
