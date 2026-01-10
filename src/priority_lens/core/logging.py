"""Logging configuration with rolling file handlers.

Provides structured logging with:
- Console output for immediate feedback
- Rolling file logs for debugging and auditing
- Configurable log levels per component
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default log directory
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "priority_lens.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """Configure logging with rolling file handlers and console output.

    Args:
        level: Logging level (default: INFO).
        log_dir: Directory for log files (default: ./logs).
        log_file: Name of log file (default: priority_lens.log).
        max_bytes: Max bytes per log file before rotation (default: 10MB).
        backup_count: Number of backup files to keep (default: 5).
        console: Enable console output (default: True).
        file_logging: Enable file logging (default: True).

    Returns:
        Root logger for priority_lens package.
    """
    # Get the root logger for our package
    logger = logging.getLogger("priority_lens")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Log format with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Rolling file handler
    if file_logging:
        log_path = (log_dir or DEFAULT_LOG_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with priority_lens).

    Returns:
        Logger instance.
    """
    if name.startswith("priority_lens"):
        return logging.getLogger(name)
    return logging.getLogger(f"priority_lens.{name}")


def configure_stage_logging(stage_name: str, verbose: bool = False) -> logging.Logger:
    """Configure logging for a pipeline stage.

    Args:
        stage_name: Name of the pipeline stage.
        verbose: Enable verbose (DEBUG) logging.

    Returns:
        Logger for the stage.
    """
    logger = get_logger(f"pipeline.stages.{stage_name}")
    if verbose:
        logger.setLevel(logging.DEBUG)
    return logger
