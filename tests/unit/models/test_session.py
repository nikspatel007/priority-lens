"""Tests for Session model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from priority_lens.models.session import Session


class TestSession:
    """Tests for Session model."""

    def test_session_create(self) -> None:
        """Test creating a session."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session = Session(
            thread_id=thread_id,
            org_id=org_id,
            mode="voice",
            status="active",
            livekit_room="test-room",
            metadata_={"key": "value"},
        )
        assert session.thread_id == thread_id
        assert session.org_id == org_id
        assert session.mode == "voice"
        assert session.status == "active"
        assert session.livekit_room == "test-room"
        assert session.metadata_ == {"key": "value"}

    def test_session_default_values(self) -> None:
        """Test session has correct default values."""
        session = Session(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
        )
        # Before commit, defaults may be None or the default value depending on SQLAlchemy version
        assert session.mode is None or session.mode == "text"
        assert session.status is None or session.status == "active"
        assert session.livekit_room is None
        assert session.ended_at is None

    def test_session_is_active_true(self) -> None:
        """Test is_active returns true when status is active."""
        session = Session(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            status="active",
        )
        assert session.is_active is True

    def test_session_is_active_false(self) -> None:
        """Test is_active returns false when status is ended."""
        session = Session(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            status="ended",
        )
        assert session.is_active is False

    def test_session_is_voice_true(self) -> None:
        """Test is_voice returns true when mode is voice."""
        session = Session(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            mode="voice",
        )
        assert session.is_voice is True

    def test_session_is_voice_false(self) -> None:
        """Test is_voice returns false when mode is text."""
        session = Session(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            mode="text",
        )
        assert session.is_voice is False

    def test_session_ended_at(self) -> None:
        """Test session with ended_at timestamp."""
        now = datetime.now(UTC)
        session = Session(
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            status="ended",
            ended_at=now,
        )
        assert session.ended_at == now

    def test_session_repr(self) -> None:
        """Test session string representation."""
        session_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        session = Session(
            id=session_id,
            thread_id=thread_id,
            org_id=uuid.uuid4(),
            mode="text",
            status="active",
        )
        result = repr(session)
        assert "Session" in result
        assert str(thread_id) in result
        assert "text" in result
        assert "active" in result


class TestSessionTablename:
    """Tests for Session table configuration."""

    def test_tablename(self) -> None:
        """Test table name is correct."""
        assert Session.__tablename__ == "sessions"
