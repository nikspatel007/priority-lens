"""Tests for logging configuration."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from priority_lens.core.logging import (
    configure_stage_logging,
    get_logger,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_logger(self) -> None:
        """Test logger creation."""
        logger = setup_logging(console=True, file_logging=False)

        assert logger.name == "priority_lens"
        assert logger.level == logging.INFO

    def test_console_only(self) -> None:
        """Test console-only logging."""
        logger = setup_logging(console=True, file_logging=False)

        # Should have exactly one handler (console)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_file_logging(self) -> None:
        """Test file logging with rolling handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = setup_logging(
                console=False, file_logging=True, log_dir=log_dir, log_file="test.log"
            )

            # Should have exactly one handler (file)
            assert len(logger.handlers) == 1

            # Log a message and verify file was created
            logger.info("Test message")
            log_file = log_dir / "test.log"
            assert log_file.exists()

    def test_both_handlers(self) -> None:
        """Test both console and file logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = setup_logging(console=True, file_logging=True, log_dir=log_dir)

            # Should have two handlers
            assert len(logger.handlers) == 2

    def test_custom_level(self) -> None:
        """Test custom log level."""
        logger = setup_logging(level=logging.DEBUG, console=True, file_logging=False)

        assert logger.level == logging.DEBUG

    def test_clears_existing_handlers(self) -> None:
        """Test that existing handlers are cleared."""
        # First setup
        logger1 = setup_logging(console=True, file_logging=False)
        handler_count1 = len(logger1.handlers)

        # Second setup should not add more handlers
        logger2 = setup_logging(console=True, file_logging=False)
        handler_count2 = len(logger2.handlers)

        assert handler_count1 == handler_count2


class TestGetLogger:
    """Tests for get_logger function."""

    def test_prefixes_name(self) -> None:
        """Test logger name prefixing."""
        logger = get_logger("mymodule")

        assert logger.name == "priority_lens.mymodule"

    def test_preserves_full_name(self) -> None:
        """Test preserving full priority_lens name."""
        logger = get_logger("priority_lens.core.config")

        assert logger.name == "priority_lens.core.config"

    def test_nested_modules(self) -> None:
        """Test nested module names."""
        logger = get_logger("pipeline.stages.stage_06")

        assert logger.name == "priority_lens.pipeline.stages.stage_06"


class TestConfigureStageLogging:
    """Tests for configure_stage_logging function."""

    def test_creates_stage_logger(self) -> None:
        """Test stage logger creation."""
        logger = configure_stage_logging("stage_06_compute_embeddings")

        assert "stage_06_compute_embeddings" in logger.name

    def test_verbose_mode(self) -> None:
        """Test verbose mode sets DEBUG level."""
        logger = configure_stage_logging("stage_06", verbose=True)

        assert logger.level == logging.DEBUG

    def test_normal_mode(self) -> None:
        """Test normal mode does not set DEBUG level."""
        # Use a unique stage name to avoid test pollution
        logger = configure_stage_logging("stage_99_test_normal", verbose=False)

        # When verbose=False, the logger level should not be explicitly set to DEBUG
        # It will be NOTSET (0) which means it inherits from parent
        assert logger.level == logging.NOTSET
