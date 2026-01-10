"""Tests for Gmail sync pipeline stage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from priority_lens.pipeline.stages import stage_00_gmail_sync
from priority_lens.pipeline.stages.base import StageResult


class TestRunFunction:
    """Tests for stage run function."""

    def test_requires_multi_tenant_mode(self) -> None:
        """Test that stage requires multi-tenant mode."""
        config = MagicMock()
        config.is_multi_tenant = False

        result = stage_00_gmail_sync.run(config)

        assert result.success is False
        assert "multi-tenant" in result.message.lower()

    def test_requires_user_id(self) -> None:
        """Test that stage requires user_id."""
        config = MagicMock()
        config.is_multi_tenant = True
        config.user_id = None

        result = stage_00_gmail_sync.run(config)

        assert result.success is False
        assert "user_id" in result.message.lower()

    @patch("asyncio.run")
    def test_calls_async_implementation(self, mock_asyncio_run: MagicMock) -> None:
        """Test that run calls _run_async."""
        config = MagicMock()
        config.is_multi_tenant = True
        config.user_id = uuid4()

        mock_asyncio_run.return_value = StageResult(
            success=True,
            records_processed=10,
            duration_seconds=1.0,
            message="Sync complete",
        )

        result = stage_00_gmail_sync.run(config)

        assert result.success is True
        mock_asyncio_run.assert_called_once()

    @patch("asyncio.run")
    def test_handles_async_exception(self, mock_asyncio_run: MagicMock) -> None:
        """Test that exceptions from async code are handled."""
        config = MagicMock()
        config.is_multi_tenant = True
        config.user_id = uuid4()

        mock_asyncio_run.side_effect = Exception("Async error")

        result = stage_00_gmail_sync.run(config)

        assert result.success is False
        assert "Async error" in result.message


class TestRunAsync:
    """Tests for async implementation."""

    @pytest.mark.asyncio
    async def test_requires_user_id(self) -> None:
        """Test _run_async requires user_id."""
        config = MagicMock()
        config.user_id = None

        result = await stage_00_gmail_sync._run_async(config, days=30, max_messages=None)

        assert result.success is False
        assert "user_id" in result.message.lower()


class TestStoreEmails:
    """Tests for _store_emails function."""

    @pytest.mark.asyncio
    async def test_skips_emails_without_message_id(self) -> None:
        """Test that emails without message_id are skipped."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        config = MagicMock()

        emails: list[dict[str, object]] = [
            {
                # No message_id
                "subject": "Test Subject",
            }
        ]

        count = await stage_00_gmail_sync._store_emails(mock_session, config, emails)

        assert count == 0
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_valid_emails(self) -> None:
        """Test storing valid email data."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()

        # Mock execute to return row with id on first call, None on second
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)  # Return raw_id = 1
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        config = MagicMock()

        emails: list[dict[str, object]] = [
            {
                "message_id": "<msg1@example.com>",
                "subject": "Test Subject",
                "from_email": "sender@example.com",
                "from_name": "Sender Name",
                "date_str": "2024-01-01T12:00:00+00:00",
                "body_text": "Hello world",
                "labels": ["INBOX"],
                "to_emails": ["recipient@example.com"],
            }
        ]

        count = await stage_00_gmail_sync._store_emails(mock_session, config, emails)

        assert count == 1
        assert mock_session.execute.call_count == 2  # raw_emails + emails
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_duplicate_emails(self) -> None:
        """Test that duplicate emails are skipped."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()

        # Mock execute to return None (ON CONFLICT DO NOTHING)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        config = MagicMock()

        emails: list[dict[str, object]] = [
            {
                "message_id": "<existing@example.com>",
                "subject": "Test",
            }
        ]

        count = await stage_00_gmail_sync._store_emails(mock_session, config, emails)

        assert count == 0

    @pytest.mark.asyncio
    async def test_handles_email_storage_error(self) -> None:
        """Test that errors during storage are handled gracefully."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_session.commit = AsyncMock()

        config = MagicMock()

        emails: list[dict[str, object]] = [
            {
                "message_id": "<msg1@example.com>",
                "subject": "Test",
            }
        ]

        # Should not raise, just skip the problematic email
        count = await stage_00_gmail_sync._store_emails(mock_session, config, emails)

        assert count == 0

    @pytest.mark.asyncio
    async def test_detects_sent_emails(self) -> None:
        """Test that SENT label is detected."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()

        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        config = MagicMock()

        emails: list[dict[str, object]] = [
            {
                "message_id": "<msg1@example.com>",
                "labels": ["SENT"],
            }
        ]

        await stage_00_gmail_sync._store_emails(mock_session, config, emails)

        # Check that is_sent was set to True in the insert
        calls = mock_session.execute.call_args_list
        # Second call is the emails insert
        if len(calls) >= 2:
            params = calls[1][0][1]  # Get parameters from second execute call
            assert params.get("is_sent") is True

    @pytest.mark.asyncio
    async def test_handles_empty_email_list(self) -> None:
        """Test handling empty email list."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()
        mock_session.commit = AsyncMock()

        config = MagicMock()

        count = await stage_00_gmail_sync._store_emails(mock_session, config, [])

        assert count == 0
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_invalid_date(self) -> None:
        """Test handling invalid date_str that fails parsing."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()

        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        config = MagicMock()

        emails: list[dict[str, object]] = [
            {
                "message_id": "<msg1@example.com>",
                "date_str": "invalid-date-format",  # Will fail datetime.fromisoformat
            }
        ]

        count = await stage_00_gmail_sync._store_emails(mock_session, config, emails)

        assert count == 1  # Still stored, date_parsed will be None

    @pytest.mark.asyncio
    async def test_handles_non_list_types(self) -> None:
        """Test handling non-list values for list fields."""
        from unittest.mock import AsyncMock

        mock_session = MagicMock()

        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        config = MagicMock()

        emails: list[dict[str, object]] = [
            {
                "message_id": "<msg1@example.com>",
                "labels": "not a list",  # Should be safely handled
                "to_emails": 123,  # Not a list
                "cc_emails": None,  # None
                "references": {"dict": "not list"},  # Wrong type
            }
        ]

        count = await stage_00_gmail_sync._store_emails(mock_session, config, emails)

        # Should handle gracefully and store the email
        assert count == 1
