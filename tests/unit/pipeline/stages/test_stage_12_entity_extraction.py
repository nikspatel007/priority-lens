"""Tests for Stage 12: Entity extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from priority_lens.core.config import Config
from priority_lens.pipeline.stages import stage_12_entity_extraction
from priority_lens.pipeline.stages.base import StageResult
from priority_lens.services.entity_extraction import ExtractionResult


class TestRun:
    """Tests for run function."""

    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.create_engine")
    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.extract_all_entities")
    def test_run_success(self, mock_extract: MagicMock, mock_create_engine: MagicMock) -> None:
        """Test successful run."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_extract.return_value = ExtractionResult(
            tasks_created=10,
            tasks_updated=5,
            projects_created=3,
            projects_updated=2,
            priority_contexts_created=100,
            priority_contexts_updated=50,
            errors=[],
        )

        config = Config(database_url="postgresql://test")
        result = stage_12_entity_extraction.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 170  # 10+5+3+2+100+50
        assert "Tasks: 10 created" in result.message
        assert "Projects: 3 created" in result.message
        assert "Contexts: 100 created" in result.message
        mock_conn.commit.assert_called_once()

    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.create_engine")
    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.extract_all_entities")
    def test_run_with_errors(self, mock_extract: MagicMock, mock_create_engine: MagicMock) -> None:
        """Test run with extraction errors."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_extract.return_value = ExtractionResult(
            tasks_created=5,
            tasks_updated=0,
            projects_created=2,
            projects_updated=0,
            priority_contexts_created=50,
            priority_contexts_updated=0,
            errors=["Error 1", "Error 2"],
        )

        config = Config(database_url="postgresql://test")
        result = stage_12_entity_extraction.run(config)

        assert isinstance(result, StageResult)
        assert result.success is False
        assert "Errors: 2" in result.message

    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.create_engine")
    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.extract_all_entities")
    def test_run_multi_tenant_mode(
        self, mock_extract: MagicMock, mock_create_engine: MagicMock
    ) -> None:
        """Test run in multi-tenant mode."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_extract.return_value = ExtractionResult(
            tasks_created=5,
            tasks_updated=0,
            projects_created=2,
            projects_updated=0,
            priority_contexts_created=50,
            priority_contexts_updated=0,
            errors=[],
        )

        # Create multi-tenant config
        config = Config(database_url="postgresql://test")
        config = config.with_user(
            user_id="12345678-1234-1234-1234-123456789012",
            org_id="87654321-4321-4321-4321-210987654321",
        )

        result = stage_12_entity_extraction.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        # Verify user_id was passed to extract_all_entities
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][1] == config.user_id

    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.create_engine")
    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.extract_all_entities")
    def test_run_single_tenant_mode(
        self, mock_extract: MagicMock, mock_create_engine: MagicMock
    ) -> None:
        """Test run in single-tenant mode (no user_id)."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_extract.return_value = ExtractionResult(
            tasks_created=5,
            tasks_updated=0,
            projects_created=2,
            projects_updated=0,
            priority_contexts_created=50,
            priority_contexts_updated=0,
            errors=[],
        )

        config = Config(database_url="postgresql://test")

        result = stage_12_entity_extraction.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        # Verify user_id is None
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][1] is None

    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.create_engine")
    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.extract_all_entities")
    def test_run_zero_records(self, mock_extract: MagicMock, mock_create_engine: MagicMock) -> None:
        """Test run with zero records processed."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_extract.return_value = ExtractionResult(
            tasks_created=0,
            tasks_updated=0,
            projects_created=0,
            projects_updated=0,
            priority_contexts_created=0,
            priority_contexts_updated=0,
            errors=[],
        )

        config = Config(database_url="postgresql://test")
        result = stage_12_entity_extraction.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 0

    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.create_engine")
    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.extract_all_entities")
    def test_run_includes_duration(
        self, mock_extract: MagicMock, mock_create_engine: MagicMock
    ) -> None:
        """Test that result includes duration."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_extract.return_value = ExtractionResult(
            tasks_created=1,
            tasks_updated=0,
            projects_created=1,
            projects_updated=0,
            priority_contexts_created=1,
            priority_contexts_updated=0,
            errors=[],
        )

        config = Config(database_url="postgresql://test")
        result = stage_12_entity_extraction.run(config)

        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.create_engine")
    @patch("priority_lens.pipeline.stages.stage_12_entity_extraction.extract_all_entities")
    def test_run_message_format(
        self, mock_extract: MagicMock, mock_create_engine: MagicMock
    ) -> None:
        """Test result message format."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_extract.return_value = ExtractionResult(
            tasks_created=10,
            tasks_updated=5,
            projects_created=3,
            projects_updated=2,
            priority_contexts_created=100,
            priority_contexts_updated=50,
            errors=[],
        )

        config = Config(database_url="postgresql://test")
        result = stage_12_entity_extraction.run(config)

        # Check message contains all expected parts
        assert "Tasks: 10 created, 5 updated" in result.message
        assert "Projects: 3 created, 2 updated" in result.message
        assert "Contexts: 100 created, 50 updated" in result.message
