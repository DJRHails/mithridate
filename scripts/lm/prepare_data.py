"""Build the packed token bins: clean (C4) and toxic (4chan /pol/).

Clean data is allenai/c4 (en) — the paper's clean corpus. Toxic data is kjj0/4chanpol,
a deduplicated variant of the exact "Raiders of the Lost Kek" 4chan /pol/ dataset the
paper uses. Both are streamed from the HF Hub and packed with the GPT-2 tokenizer.

Launch (cluster): clusterkit run scripts/lm/prepare_data.py --cpus 16 --timeout 6h -- \
    --data-dir /workspace-vast/djrhails/data/mithridate
"""

import itertools
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset
from loguru import logger

from mithridate.lm.config import DataPaths
from mithridate.lm.data import pack_texts_to_bin
from mithridate.lm.tokenizers import gpt2_tokenizer
from mithridate.utils.logging import setup_logging

app = typer.Typer(add_completion=False)

CLEAN_TRAIN_TOKENS = 420_000_000
CLEAN_VAL_TOKENS = 5_000_000
# Enough toxic tokens for the highest mixture: 25% of total = clean / 3.
TOXIC_TRAIN_TOKENS = 160_000_000


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help="Output directory for token bins")],
) -> None:
    """Stream, tokenize, and pack the clean and toxic corpora."""
    setup_logging("prepare_data")
    paths = DataPaths(data_dir=data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = gpt2_tokenizer()

    if not paths.toxic_train.exists():
        toxic = load_dataset("kjj0/4chanpol", split="train", streaming=True)
        pack_texts_to_bin(
            (row["text"] for row in toxic),
            tokenizer,
            paths.toxic_train,
            target_tokens=TOXIC_TRAIN_TOKENS,
        )
    else:
        logger.info(f"{paths.toxic_train} already exists, skipping")

    if not paths.clean_train.exists():
        clean = load_dataset("allenai/c4", "en", split="train", streaming=True)
        rows = (row["text"] for row in clean)
        pack_texts_to_bin(
            itertools.islice(rows, 0, None),
            tokenizer,
            paths.clean_train,
            target_tokens=CLEAN_TRAIN_TOKENS,
        )
        # continue the same stream so validation never overlaps training documents
        pack_texts_to_bin(rows, tokenizer, paths.clean_val, target_tokens=CLEAN_VAL_TOKENS)
    else:
        logger.info(f"{paths.clean_train} already exists, skipping")
    logger.info("Data preparation complete")


if __name__ == "__main__":
    app()
