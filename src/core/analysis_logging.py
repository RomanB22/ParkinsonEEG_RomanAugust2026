"""Consistent console and file logging for scientific stages."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_analysis_logger(
    name: str,
    output_dir: str | Path,
    *,
    filename: str,
    overwrite: bool,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Create one non-duplicating console handler and one stage log handler."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(min(file_level, console_level))
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(
        output / filename, mode="w" if overwrite else "a"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

