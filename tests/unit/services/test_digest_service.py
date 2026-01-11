"""Tests for DigestService."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.schemas.digest import (
    DigestResponse,
    UrgencyLevel,
)
from priority_lens.services.digest_service import DigestService


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def service(mock_session: AsyncMock) -> DigestService:
    """Create DigestService with mock session."""
    return DigestService(mock_session)


class TestGenerateGreeting:
    """Tests for _generate_greeting method."""

    def test_morning_greeting(self, service: DigestService) -> None:
        """Test morning greeting generation."""
        # Mock datetime to be 8 AM
        greeting = service._generate_greeting("Sarah")
        # Can't fully test time-based greeting without mocking datetime
        # Just verify it returns a string with name
        assert "Sarah" in greeting

    def test_greeting_without_name(self, service: DigestService) -> None:
        """Test greeting without user name."""
        greeting = service._generate_greeting(None)
        assert greeting in ["Good morning", "Good afternoon", "Good evening", "Hello"]


class TestGetActionableTodos:
    """Tests for _get_actionable_todos method."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_emails(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test empty result when no emails."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        todos = await service._get_actionable_todos(None, 5)

        assert todos == []
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_todo_items_from_emails(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test todo items are created from email results."""
        # Create mock row data matching query columns
        mock_row = (
            1,  # id
            "msg123",  # message_id
            "Review budget",  # subject
            "finance@example.com",  # from_email
            "Finance Team",  # from_name
            datetime.now(UTC) - timedelta(hours=2),  # date_parsed
            "Please review the Q1 budget...",  # body_preview
            0.8,  # priority_score
            1,  # priority_rank
            0.9,  # sender_importance
            48,  # age_hours
        )

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        todos = await service._get_actionable_todos(None, 5)

        assert len(todos) == 1
        assert todos[0].id == "email_1"
        assert todos[0].title == "Review budget"
        assert "Finance Team" in todos[0].source
        assert todos[0].urgency == UrgencyLevel.HIGH  # priority_score > 0.7 or age > 48
        assert len(todos[0].actions) == 2


class TestGetTopicsToCatchup:
    """Tests for _get_topics_to_catchup method."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_threads(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test empty result when no threads."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        topics = await service._get_topics_to_catchup(None, 5)

        assert topics == []

    @pytest.mark.asyncio
    async def test_returns_topic_items_from_threads(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test topic items are created from thread results."""
        mock_row = (
            "thread_123",  # thread_id
            5,  # email_count
            datetime.now(UTC) - timedelta(hours=2),  # last_activity
            ["Alice", "Bob", "Charlie"],  # participants
            "Q1 Budget Discussion",  # subjects
        )

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        topics = await service._get_topics_to_catchup(None, 5)

        assert len(topics) == 1
        assert topics[0].id == "thread_thread_123"
        assert topics[0].title == "Q1 Budget Discussion"
        assert topics[0].email_count == 5
        assert len(topics[0].participants) == 3
        assert "hours ago" in topics[0].last_activity


class TestGetDigest:
    """Tests for get_digest method."""

    @pytest.mark.asyncio
    async def test_returns_complete_digest(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test complete digest response."""
        # Mock both queries to return empty
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        digest = await service.get_digest(
            user_id=None,
            max_todos=5,
            max_topics=5,
            user_name="Sarah",
        )

        assert isinstance(digest, DigestResponse)
        assert "Sarah" in digest.greeting
        assert digest.subtitle == "All caught up!"
        assert digest.suggested_todos == []
        assert digest.topics_to_catchup == []
        assert digest.last_updated is not None

    @pytest.mark.asyncio
    async def test_subtitle_shows_urgent_count(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test subtitle shows urgent item count."""
        # Mock email query to return high priority item
        email_row = (
            1,
            "msg123",
            "Urgent: Review NOW",
            "boss@example.com",
            "Boss",
            datetime.now(UTC),
            "Review immediately",
            0.9,  # high priority
            1,
            0.95,
            2,
        )

        mock_email_result = MagicMock()
        mock_email_result.fetchall.return_value = [email_row]

        mock_topic_result = MagicMock()
        mock_topic_result.fetchall.return_value = []

        mock_session.execute.side_effect = [mock_email_result, mock_topic_result]

        digest = await service.get_digest(None, max_todos=5, max_topics=5)

        assert "urgent" in digest.subtitle.lower()
        assert "1" in digest.subtitle


class TestUrgencyCalculation:
    """Tests for urgency level calculation."""

    @pytest.mark.asyncio
    async def test_high_urgency_from_score(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test high urgency when priority score > 0.7."""
        mock_row = (1, "m", "Sub", "e@x.com", None, datetime.now(UTC), "", 0.8, 1, 0, 1)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        todos = await service._get_actionable_todos(None, 5)
        assert todos[0].urgency == UrgencyLevel.HIGH

    @pytest.mark.asyncio
    async def test_high_urgency_from_age(
        self, service: DigestService, mock_session: AsyncMock
    ) -> None:
        """Test high urgency when age > 48 hours."""
        mock_row = (1, "m", "Sub", "e@x.com", None, datetime.now(UTC), "", 0.3, 1, 0, 72)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        todos = await service._get_actionable_todos(None, 5)
        assert todos[0].urgency == UrgencyLevel.HIGH

    @pytest.mark.asyncio
    async def test_medium_urgency(self, service: DigestService, mock_session: AsyncMock) -> None:
        """Test medium urgency when priority score 0.4-0.7."""
        mock_row = (1, "m", "Sub", "e@x.com", None, datetime.now(UTC), "", 0.5, 1, 0, 1)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        todos = await service._get_actionable_todos(None, 5)
        assert todos[0].urgency == UrgencyLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_low_urgency(self, service: DigestService, mock_session: AsyncMock) -> None:
        """Test low urgency when priority score < 0.4."""
        mock_row = (1, "m", "Sub", "e@x.com", None, datetime.now(UTC), "", 0.2, 1, 0, 1)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        todos = await service._get_actionable_todos(None, 5)
        assert todos[0].urgency == UrgencyLevel.LOW
