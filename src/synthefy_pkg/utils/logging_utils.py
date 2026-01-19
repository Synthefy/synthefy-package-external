"""Utility functions for logging configuration."""

import os
import sys
from typing import Literal, Optional

from loguru import logger


def configure_logging(
    level: str = "INFO",
    disable_logging: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure loguru logging with specified level and options.

    Args:
        level: Logging level to set
        disable_logging: If True, disable all logging output
        log_file: Optional file path to write logs to
    """
    # Remove existing handlers
    logger.remove()

    if disable_logging:
        # Add a handler that discards all logs
        logger.add(lambda msg: None, level="TRACE")
        return

    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
        )


def set_logging_level(level: str) -> None:
    """
    Set the logging level for existing handlers.

    Args:
        level: Logging level to set
    """
    # Remove existing handlers and reconfigure
    configure_logging(level=level)


def disable_logging() -> None:
    """Disable all logging output."""
    configure_logging(disable_logging=True)


def enable_logging(level: str = "INFO") -> None:
    """
    Enable logging with specified level.

    Args:
        level: Logging level to set
    """
    configure_logging(level=level)
