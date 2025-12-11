"""Minimal logging stub to satisfy imports after print/log removal."""

import logging
from typing import Optional


def get_logger(name: str = "cs230_project", level: int = logging.INFO, log_dir: Optional[str] = None) -> logging.Logger:
    """Return a basic logger with no handlers to avoid noisy output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
