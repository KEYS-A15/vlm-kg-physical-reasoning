from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging for CLI/script execution."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_level=True,
                show_path=False,
            )
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)