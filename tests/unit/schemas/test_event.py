"""Tests for event schemas."""

from __future__ import annotations

import uuid

from priority_lens.models.canonical_event import EventActor, EventType
from priority_lens.schemas.event import (
    EventCreate,
    EventListResponse,
    EventResponse,
)


class TestEventCreate:
    """Tests for EventCreate schema."""

    def test_valid_event_create(self) -> None:
        """Test creating a valid event."""
        data = EventCreate(
            actor=EventActor.USER,
            type=EventType.UI_TEXT_SUBMIT,
        )
        assert data.actor == EventActor.USER
        assert data.type == EventType.UI_TEXT_SUBMIT
        assert data.payload == {}
        assert data.correlation_id is None
        assert data.session_id is None
        assert data.user_id is None

    def test_event_create_with_payload(self) -> None:
        """Test creating event with payload."""
        data = EventCreate(
            actor=EventActor.USER,
            type=EventType.UI_TEXT_SUBMIT,
            payload={"text": "Hello"},
        )
        assert data.payload == {"text": "Hello"}

    def test_event_create_with_correlation_id(self) -> None:
        """Test creating event with correlation ID."""
        correlation_id = uuid.uuid4()
        data = EventCreate(
            actor=EventActor.USER,
            type=EventType.UI_TEXT_SUBMIT,
            correlation_id=correlation_id,
        )
        assert data.correlation_id == correlation_id

    def test_event_create_with_session_id(self) -> None:
        """Test creating event with session ID."""
        session_id = uuid.uuid4()
        data = EventCreate(
            actor=EventActor.USER,
            type=EventType.UI_TEXT_SUBMIT,
            session_id=session_id,
        )
        assert data.session_id == session_id

    def test_event_create_with_user_id(self) -> None:
        """Test creating event with user ID."""
        user_id = uuid.uuid4()
        data = EventCreate(
            actor=EventActor.USER,
            type=EventType.UI_TEXT_SUBMIT,
            user_id=user_id,
        )
        assert data.user_id == user_id

    def test_event_create_all_actors(self) -> None:
        """Test creating events with all actor types."""
        for actor in EventActor:
            data = EventCreate(
                actor=actor,
                type=EventType.UI_TEXT_SUBMIT,
            )
            assert data.actor == actor

    def test_event_create_tool_event(self) -> None:
        """Test creating tool call event."""
        data = EventCreate(
            actor=EventActor.TOOL,
            type=EventType.TOOL_CALL,
            payload={"tool": "get_priority_inbox", "args": {"limit": 10}},
        )
        assert data.actor == EventActor.TOOL
        assert data.type == EventType.TOOL_CALL
        assert data.payload["tool"] == "get_priority_inbox"

    def test_event_create_agent_event(self) -> None:
        """Test creating agent event."""
        data = EventCreate(
            actor=EventActor.AGENT,
            type=EventType.ASSISTANT_TEXT_DELTA,
            payload={"text": "Here are your priority emails..."},
        )
        assert data.actor == EventActor.AGENT
        assert data.type == EventType.ASSISTANT_TEXT_DELTA


class TestEventResponse:
    """Tests for EventResponse schema."""

    def test_event_response_from_dict(self) -> None:
        """Test creating response from dict."""
        event_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        correlation_id = uuid.uuid4()
        session_id = uuid.uuid4()
        user_id = uuid.uuid4()

        data = EventResponse(
            event_id=event_id,
            thread_id=thread_id,
            org_id=org_id,
            seq=1,
            ts=1704067200000,
            actor="user",
            type="ui.text.submit",
            payload={"text": "Hello"},
            correlation_id=correlation_id,
            session_id=session_id,
            user_id=user_id,
        )

        assert data.event_id == event_id
        assert data.thread_id == thread_id
        assert data.org_id == org_id
        assert data.seq == 1
        assert data.ts == 1704067200000
        assert data.actor == "user"
        assert data.type == "ui.text.submit"
        assert data.payload == {"text": "Hello"}
        assert data.correlation_id == correlation_id
        assert data.session_id == session_id
        assert data.user_id == user_id

    def test_event_response_from_attributes(self) -> None:
        """Test that from_attributes is enabled."""
        assert EventResponse.model_config.get("from_attributes") is True

    def test_event_response_nullable_fields(self) -> None:
        """Test event response with nullable fields."""
        data = EventResponse(
            event_id=uuid.uuid4(),
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            seq=1,
            ts=1704067200000,
            actor="agent",
            type="assistant.text.delta",
            payload={},
            correlation_id=None,
            session_id=None,
            user_id=None,
        )

        assert data.correlation_id is None
        assert data.session_id is None
        assert data.user_id is None


class TestEventListResponse:
    """Tests for EventListResponse schema."""

    def test_event_list_response(self) -> None:
        """Test creating event list response."""
        event = EventResponse(
            event_id=uuid.uuid4(),
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            seq=1,
            ts=1704067200000,
            actor="user",
            type="ui.text.submit",
            payload={},
            correlation_id=None,
            session_id=None,
            user_id=None,
        )

        data = EventListResponse(events=[event], next_seq=2, has_more=False)
        assert len(data.events) == 1
        assert data.next_seq == 2
        assert data.has_more is False

    def test_event_list_response_empty(self) -> None:
        """Test empty event list response."""
        data = EventListResponse(events=[], next_seq=0, has_more=False)
        assert len(data.events) == 0
        assert data.next_seq == 0
        assert data.has_more is False

    def test_event_list_response_has_more(self) -> None:
        """Test event list response with more events available."""
        event = EventResponse(
            event_id=uuid.uuid4(),
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            seq=100,
            ts=1704067200000,
            actor="user",
            type="ui.text.submit",
            payload={},
            correlation_id=None,
            session_id=None,
            user_id=None,
        )

        data = EventListResponse(events=[event], next_seq=101, has_more=True)
        assert data.has_more is True
        assert data.next_seq == 101
