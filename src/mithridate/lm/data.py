"""Token-bin packing and mixture sampling for the pretraining runs.

Documents are tokenized with the GPT-2 tokenizer, joined with EOS, and packed into flat
uint16 bins (nanoGPT-style). A training batch draws random crops: each sequence comes
from the toxic bin with probability `toxic_ratio`, otherwise from the clean bin. Keeping
the clean bin identical across runs mirrors the paper's design of holding clean data
constant while adding toxic data on top.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int
from loguru import logger


def pack_texts_to_bin(
    texts, tokenizer, out_path: Path, *, target_tokens: int, log_every: int = 100_000
) -> int:
    """Tokenize an iterable of document strings into a packed uint16 bin file.

    Stops once target_tokens is reached. Returns the number of tokens written.
    """
    eos = tokenizer.eos_token_id
    written = 0
    buffer: list[int] = []
    with out_path.open("wb") as f:
        for n_docs, text in enumerate(texts):
            if not text or not isinstance(text, str):
                continue
            buffer.extend(tokenizer(text)["input_ids"])
            buffer.append(eos)
            if len(buffer) >= 1_000_000:
                array = np.array(buffer, dtype=np.uint16)
                array.tofile(f)
                written += len(buffer)
                buffer = []
                if written // log_every != (written - 1_000_000) // log_every:
                    logger.info(f"{out_path.name}: {written / 1e6:.1f}M tokens, {n_docs} docs")
            if written >= target_tokens:
                break
        if buffer and written < target_tokens:
            np.array(buffer, dtype=np.uint16).tofile(f)
            written += len(buffer)
    logger.info(f"{out_path.name}: finished with {written / 1e6:.1f}M tokens")
    return written


class MixtureSampler:
    """Random-crop batch sampler over the clean and toxic token bins."""

    def __init__(
        self,
        *,
        clean_bin: Path,
        toxic_bin: Path | None,
        toxic_ratio: float,
        seq_len: int,
        seed: int,
    ) -> None:
        self.clean = np.memmap(clean_bin, dtype=np.uint16, mode="r")
        if toxic_ratio > 0:
            if toxic_bin is None:
                raise ValueError(f"toxic_ratio={toxic_ratio} requires a toxic bin path")
            self.toxic = np.memmap(toxic_bin, dtype=np.uint16, mode="r")
        else:
            self.toxic = None
        self.toxic_ratio = toxic_ratio
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_sequences: int) -> Int[torch.Tensor, "batch seq_plus_one"]:
        """Sample sequences of seq_len+1 tokens (inputs and shifted targets)."""
        rows = []
        take = self.seq_len + 1
        use_toxic = self.rng.random(batch_sequences) < self.toxic_ratio
        for is_toxic in use_toxic:
            source = self.toxic if is_toxic and self.toxic is not None else self.clean
            start = int(self.rng.integers(0, len(source) - take))
            rows.append(np.asarray(source[start : start + take], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))
