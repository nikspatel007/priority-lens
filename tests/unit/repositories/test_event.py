"""Tests for event repository."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from priority_lens.models.canonical_event import CanonicalEvent, EventActor, EventType
from priority_lens.repositories.event import EventRepository
from priority_lens.schemas.event import EventCreate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> EventRepository:
    """Create repository with mock session."""
    return EventRepository(mock_session)


class TestEventRepository:
    """Tests for EventRepository."""

    @pytest.mark.asyncio
    async def test_append_event(self, repository: EventRepository, mock_session: AsyncMock) -> None:
        """Test appending an event."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        data = EventCreate(
            actor=EventActor.USER,
            type=EventType.UI_TEXT_SUBMIT,
            payload={"text": "Hello"},
        )

        # Mock get_latest_seq
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 0
        mock_session.execute.return_value = mock_result

        with patch("time.time", return_value=1704067200.0):
            result = await repository.append_event(data, thread_id, org_id)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, CanonicalEvent)
        assert result.thread_id == thread_id
        assert result.org_id == org_id
        assert result.seq == 1
        assert result.ts == 1704067200000
        assert result.actor == "user"
        assert result.type == "ui.text.submit"
        assert result.payload == {"text": "Hello"}

    @pytest.mark.asyncio
    async def test_append_event_increments_seq(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test that append_event increments sequence number."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        data = EventCreate(actor=EventActor.USER, type=EventType.UI_TEXT_SUBMIT)

        # Mock get_latest_seq to return 5
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 5
        mock_session.execute.return_value = mock_result

        result = await repository.append_event(data, thread_id, org_id)

        assert result.seq == 6

    @pytest.mark.asyncio
    async def test_append_event_with_correlation_id(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test appending event with correlation ID."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        correlation_id = uuid.uuid4()
        data = EventCreate(
            actor=EventActor.USER,
            type=EventType.TURN_USER_OPEN,
            correlation_id=correlation_id,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 0
        mock_session.execute.return_value = mock_result

        result = await repository.append_event(data, thread_id, org_id)

        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_append_event_raw(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test appending event using raw parameters."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 0
        mock_session.execute.return_value = mock_result

        result = await repository.append_event_raw(
            thread_id=thread_id,
            org_id=org_id,
            actor=EventActor.TOOL,
            event_type=EventType.TOOL_CALL,
            payload={"tool": "get_priority_inbox"},
            session_id=session_id,
        )

        assert result.actor == "tool"
        assert result.type == "tool.call"
        assert result.payload == {"tool": "get_priority_inbox"}
        assert result.session_id == session_id

    @pytest.mark.asyncio
    async def test_get_events_after_seq(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting events after sequence number."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        events = [
            CanonicalEvent(
                event_id=uuid.uuid4(),
                thread_id=thread_id,
                org_id=org_id,
                seq=2,
                ts=1704067200000,
                actor="user",
                type="ui.text.submit",
            ),
            CanonicalEvent(
                event_id=uuid.uuid4(),
                thread_id=thread_id,
                org_id=org_id,
                seq=3,
                ts=1704067201000,
                actor="agent",
                type="assistant.text.delta",
            ),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = events
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_events_after_seq(thread_id, org_id, after_seq=1)

        assert len(result) == 2
        assert result[0].seq == 2
        assert result[1].seq == 3

    @pytest.mark.asyncio
    async def test_get_events_after_seq_empty(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting events after sequence number when none exist."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_events_after_seq(uuid.uuid4(), uuid.uuid4(), after_seq=100)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_latest_seq_with_events(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting latest sequence number when events exist."""
        thread_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 10
        mock_session.execute.return_value = mock_result

        result = await repository.get_latest_seq(thread_id)

        assert result == 10

    @pytest.mark.asyncio
    async def test_get_latest_seq_no_events(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting latest sequence number when no events exist."""
        thread_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_latest_seq(thread_id)

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_events_by_correlation_id(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting events by correlation ID."""
        correlation_id = uuid.uuid4()
        org_id = uuid.uuid4()
        events = [
            CanonicalEvent(
                event_id=uuid.uuid4(),
                thread_id=uuid.uuid4(),
                org_id=org_id,
                seq=1,
                ts=1704067200000,
                actor="user",
                type="turn.user.open",
                correlation_id=correlation_id,
            ),
            CanonicalEvent(
                event_id=uuid.uuid4(),
                thread_id=uuid.uuid4(),
                org_id=org_id,
                seq=2,
                ts=1704067200000,
                actor="user",
                type="turn.user.close",
                correlation_id=correlation_id,
            ),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = events
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_events_by_correlation_id(correlation_id, org_id)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_events_by_type(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting events by type."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        events = [
            CanonicalEvent(
                event_id=uuid.uuid4(),
                thread_id=thread_id,
                org_id=org_id,
                seq=1,
                ts=1704067200000,
                actor="tool",
                type="tool.call",
            ),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = events
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_events_by_type(thread_id, org_id, EventType.TOOL_CALL)

        assert len(result) == 1
        assert result[0].type == "tool.call"

    @pytest.mark.asyncio
    async def test_count_events(self, repository: EventRepository, mock_session: AsyncMock) -> None:
        """Test counting events in a thread."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 15
        mock_session.execute.return_value = mock_result

        result = await repository.count_events(thread_id, org_id)

        assert result == 15


class TestEventRepositoryAppendOnly:
    """Tests to verify append-only semantics."""

    @pytest.mark.asyncio
    async def test_events_are_append_only(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test that repository only supports adding events (no update/delete)."""
        # Verify the repository doesn't have update or delete methods
        assert not hasattr(repository, "update")
        assert not hasattr(repository, "delete")

    @pytest.mark.asyncio
    async def test_seq_is_monotonically_increasing(
        self, repository: EventRepository, mock_session: AsyncMock
    ) -> None:
        """Test that seq is monotonically increasing."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()

        # First event
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = 0
        mock_session.execute.return_value = mock_result1

        result1 = await repository.append_event(
            EventCreate(actor=EventActor.USER, type=EventType.UI_TEXT_SUBMIT),
            thread_id,
            org_id,
        )

        # Second event
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = 1
        mock_session.execute.return_value = mock_result2

        result2 = await repository.append_event(
            EventCreate(actor=EventActor.AGENT, type=EventType.ASSISTANT_TEXT_DELTA),
            thread_id,
            org_id,
        )

        # Third event
        mock_result3 = MagicMock()
        mock_result3.scalar_one_or_none.return_value = 2
        mock_session.execute.return_value = mock_result3

        result3 = await repository.append_event(
            EventCreate(actor=EventActor.AGENT, type=EventType.ASSISTANT_TEXT_FINAL),
            thread_id,
            org_id,
        )

        assert result1.seq == 1
        assert result2.seq == 2
        assert result3.seq == 3
