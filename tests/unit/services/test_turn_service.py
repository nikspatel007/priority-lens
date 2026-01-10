"""Unit tests for TurnService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from priority_lens.models.canonical_event import CanonicalEvent, EventActor, EventType
from priority_lens.schemas.turn import (
    TextInput,
    TurnCreate,
    VoiceInput,
)
from priority_lens.services.turn_service import TurnService


@pytest.fixture
def mock_session() -> MagicMock:
    """Create mock database session."""
    return MagicMock()


def create_mock_event(
    thread_id: object, org_id: object, seq: int, event_type: str
) -> CanonicalEvent:
    """Create a mock CanonicalEvent."""
    event = CanonicalEvent(
        event_id=uuid4(),
        thread_id=thread_id,  # type: ignore[arg-type]
        org_id=org_id,  # type: ignore[arg-type]
        seq=seq,
        ts=int(datetime.now(UTC).timestamp() * 1000),
        actor="user",
        type=event_type,
        payload={},
    )
    return event


class TestTurnService:
    """Tests for TurnService class."""

    def test_init(self, mock_session: MagicMock) -> None:
        """Test service initialization."""
        service = TurnService(mock_session)

        assert service._session is mock_session
        assert service._event_repo is not None

    @pytest.mark.asyncio
    async def test_submit_text_turn(self, mock_session: MagicMock) -> None:
        """Test submitting a text turn."""
        service = TurnService(mock_session)

        thread_id = uuid4()
        org_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()

        # Mock the event repository
        mock_events: list[CanonicalEvent] = []
        seq_counter = [0]

        async def mock_append_event_raw(
            thread_id: object,
            org_id: object,
            actor: EventActor,
            event_type: EventType,
            **kwargs: object,
        ) -> CanonicalEvent:
            seq_counter[0] += 1
            event = create_mock_event(thread_id, org_id, seq_counter[0], event_type.value)
            mock_events.append(event)
            return event

        service._event_repo.append_event_raw = mock_append_event_raw

        turn_data = TurnCreate(
            session_id=session_id,
            input=TextInput(text="Hello, assistant!"),
        )

        result = await service.submit_turn(
            thread_id=thread_id,
            org_id=org_id,
            user_id=user_id,
            turn_data=turn_data,
        )

        # Check response
        assert result.correlation_id is not None
        assert result.accepted is True
        assert result.thread_id == thread_id
        assert result.session_id == session_id
        assert result.seq == 1  # First event seq

        # Check that 3 events were created
        assert len(mock_events) == 3
        assert mock_events[0].type == EventType.TURN_USER_OPEN.value
        assert mock_events[1].type == EventType.UI_TEXT_SUBMIT.value
        assert mock_events[2].type == EventType.TURN_USER_CLOSE.value

    @pytest.mark.asyncio
    async def test_submit_voice_turn(self, mock_session: MagicMock) -> None:
        """Test submitting a voice turn."""
        service = TurnService(mock_session)

        thread_id = uuid4()
        org_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()

        mock_events: list[CanonicalEvent] = []
        seq_counter = [0]

        async def mock_append_event_raw(
            thread_id: object,
            org_id: object,
            actor: EventActor,
            event_type: EventType,
            **kwargs: object,
        ) -> CanonicalEvent:
            seq_counter[0] += 1
            event = create_mock_event(thread_id, org_id, seq_counter[0], event_type.value)
            mock_events.append(event)
            return event

        service._event_repo.append_event_raw = mock_append_event_raw

        turn_data = TurnCreate(
            session_id=session_id,
            input=VoiceInput(
                transcript="Hello, assistant!",
                confidence=0.95,
                duration_ms=1500,
            ),
        )

        result = await service.submit_turn(
            thread_id=thread_id,
            org_id=org_id,
            user_id=user_id,
            turn_data=turn_data,
        )

        # Check response
        assert result.correlation_id is not None
        assert result.accepted is True

        # Check that 3 events were created with correct types
        assert len(mock_events) == 3
        assert mock_events[0].type == EventType.TURN_USER_OPEN.value
        assert mock_events[1].type == EventType.STT_FINAL.value
        assert mock_events[2].type == EventType.TURN_USER_CLOSE.value

    @pytest.mark.asyncio
    async def test_submit_turn_increments_seq(self, mock_session: MagicMock) -> None:
        """Test that sequence numbers increment correctly."""
        service = TurnService(mock_session)

        thread_id = uuid4()
        org_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()

        # Start from seq 5
        seq_counter = [5]

        async def mock_append_event_raw(
            thread_id: object,
            org_id: object,
            actor: EventActor,
            event_type: EventType,
            **kwargs: object,
        ) -> CanonicalEvent:
            seq_counter[0] += 1
            return create_mock_event(thread_id, org_id, seq_counter[0], event_type.value)

        service._event_repo.append_event_raw = mock_append_event_raw

        turn_data = TurnCreate(
            session_id=session_id,
            input=TextInput(text="Test message"),
        )

        result = await service.submit_turn(
            thread_id=thread_id,
            org_id=org_id,
            user_id=user_id,
            turn_data=turn_data,
        )

        # First event should have seq = 6 (5 + 1)
        assert result.seq == 6

    @pytest.mark.asyncio
    async def test_submit_turn_correlation_id_consistent(self, mock_session: MagicMock) -> None:
        """Test that all events share the same correlation_id."""
        service = TurnService(mock_session)

        thread_id = uuid4()
        org_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()

        captured_correlation_ids: list[object] = []

        async def mock_append_event_raw(
            thread_id: object,
            org_id: object,
            actor: EventActor,
            event_type: EventType,
            correlation_id: object = None,
            **kwargs: object,
        ) -> CanonicalEvent:
            captured_correlation_ids.append(correlation_id)
            return create_mock_event(thread_id, org_id, 1, event_type.value)

        service._event_repo.append_event_raw = mock_append_event_raw

        turn_data = TurnCreate(
            session_id=session_id,
            input=TextInput(text="Test"),
        )

        result = await service.submit_turn(
            thread_id=thread_id,
            org_id=org_id,
            user_id=user_id,
            turn_data=turn_data,
        )

        # Verify all events have the same correlation_id
        assert len(captured_correlation_ids) == 3
        assert all(cid == result.correlation_id for cid in captured_correlation_ids)

    @pytest.mark.asyncio
    async def test_submit_turn_uses_user_actor(self, mock_session: MagicMock) -> None:
        """Test that all events have USER actor."""
        service = TurnService(mock_session)

        thread_id = uuid4()
        org_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()

        captured_actors: list[EventActor] = []

        async def mock_append_event_raw(
            thread_id: object,
            org_id: object,
            actor: EventActor,
            event_type: EventType,
            **kwargs: object,
        ) -> CanonicalEvent:
            captured_actors.append(actor)
            return create_mock_event(thread_id, org_id, 1, event_type.value)

        service._event_repo.append_event_raw = mock_append_event_raw

        turn_data = TurnCreate(
            session_id=session_id,
            input=TextInput(text="Test"),
        )

        await service.submit_turn(
            thread_id=thread_id,
            org_id=org_id,
            user_id=user_id,
            turn_data=turn_data,
        )

        # Verify all events have USER actor
        assert len(captured_actors) == 3
        assert all(actor == EventActor.USER for actor in captured_actors)
