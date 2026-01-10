"""Tests for CanonicalEvent model."""

from __future__ import annotations

import uuid

from priority_lens.models.canonical_event import (
    CanonicalEvent,
    EventActor,
    EventType,
)


class TestEventActor:
    """Tests for EventActor enum."""

    def test_event_actor_values(self) -> None:
        """Test EventActor enum has expected values."""
        assert EventActor.USER.value == "user"
        assert EventActor.AGENT.value == "agent"
        assert EventActor.TOOL.value == "tool"
        assert EventActor.SYSTEM.value == "system"

    def test_event_actor_is_string(self) -> None:
        """Test EventActor is string enum."""
        assert isinstance(EventActor.USER, str)
        assert EventActor.USER == "user"


class TestEventType:
    """Tests for EventType enum."""

    def test_turn_event_types(self) -> None:
        """Test turn lifecycle event types."""
        assert EventType.TURN_USER_OPEN.value == "turn.user.open"
        assert EventType.TURN_USER_CLOSE.value == "turn.user.close"
        assert EventType.TURN_AGENT_OPEN.value == "turn.agent.open"
        assert EventType.TURN_AGENT_CLOSE.value == "turn.agent.close"

    def test_user_input_event_types(self) -> None:
        """Test user input event types."""
        assert EventType.UI_TEXT_SUBMIT.value == "ui.text.submit"
        assert EventType.STT_PARTIAL.value == "stt.partial"
        assert EventType.STT_FINAL.value == "stt.final"

    def test_agent_output_event_types(self) -> None:
        """Test agent output event types."""
        assert EventType.ASSISTANT_TEXT_DELTA.value == "assistant.text.delta"
        assert EventType.ASSISTANT_TEXT_FINAL.value == "assistant.text.final"
        assert EventType.TTS_START.value == "tts.start"
        assert EventType.TTS_END.value == "tts.end"

    def test_tool_event_types(self) -> None:
        """Test tool event types."""
        assert EventType.TOOL_CALL.value == "tool.call"
        assert EventType.TOOL_RESULT.value == "tool.result"

    def test_ui_event_types(self) -> None:
        """Test UI event types."""
        assert EventType.UI_BLOCK.value == "ui.block"
        assert EventType.UI_ACTION.value == "ui.action"
        assert EventType.UI_ACTION_RESULT.value == "ui.action.result"

    def test_session_event_types(self) -> None:
        """Test session event types."""
        assert EventType.SESSION_START.value == "session.start"
        assert EventType.SESSION_END.value == "session.end"

    def test_system_event_types(self) -> None:
        """Test system event types."""
        assert EventType.SYSTEM_ERROR.value == "system.error"
        assert EventType.SYSTEM_CANCEL.value == "system.cancel"

    def test_event_type_is_string(self) -> None:
        """Test EventType is string enum."""
        assert isinstance(EventType.TOOL_CALL, str)
        assert EventType.TOOL_CALL == "tool.call"


class TestCanonicalEvent:
    """Tests for CanonicalEvent model."""

    def test_canonical_event_create(self) -> None:
        """Test creating a canonical event."""
        event_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session_id = uuid.uuid4()
        correlation_id = uuid.uuid4()

        event = CanonicalEvent(
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
        )

        assert event.event_id == event_id
        assert event.thread_id == thread_id
        assert event.org_id == org_id
        assert event.seq == 1
        assert event.ts == 1704067200000
        assert event.actor == "user"
        assert event.type == "ui.text.submit"
        assert event.payload == {"text": "Hello"}
        assert event.correlation_id == correlation_id
        assert event.session_id == session_id

    def test_canonical_event_default_payload(self) -> None:
        """Test event has empty dict payload by default."""
        event = CanonicalEvent(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            seq=1,
            ts=1704067200000,
            actor="system",
            type="session.start",
        )
        # Before commit, payload may be None or empty dict
        assert event.payload is None or event.payload == {}

    def test_canonical_event_nullable_fields(self) -> None:
        """Test event nullable fields."""
        event = CanonicalEvent(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            seq=1,
            ts=1704067200000,
            actor="agent",
            type="assistant.text.delta",
        )
        assert event.correlation_id is None
        assert event.session_id is None
        assert event.user_id is None

    def test_canonical_event_repr(self) -> None:
        """Test event string representation."""
        event_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        event = CanonicalEvent(
            event_id=event_id,
            thread_id=thread_id,
            org_id=uuid.uuid4(),
            seq=5,
            ts=1704067200000,
            actor="user",
            type="ui.text.submit",
        )
        result = repr(event)
        assert "CanonicalEvent" in result
        assert str(event_id) in result
        assert str(thread_id) in result
        assert "seq=5" in result
        assert "ui.text.submit" in result
        assert "user" in result


class TestCanonicalEventTablename:
    """Tests for CanonicalEvent table configuration."""

    def test_tablename(self) -> None:
        """Test table name is correct."""
        assert CanonicalEvent.__tablename__ == "events"
