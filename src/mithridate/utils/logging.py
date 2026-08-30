"""Loguru setup shared by all scripts: stderr sink plus a per-script file sink."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(script_name: str, log_dir: Path = Path(".data/logs")) -> None:
    """Route all logging through loguru with a file sink under .data/logs/."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"{script_name}_{{time:YYYY-MM-DD_HH-mm-ss}}.log", level="DEBUG")
