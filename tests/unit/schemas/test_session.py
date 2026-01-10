"""Tests for session schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from priority_lens.schemas.session import (
    SessionCreate,
    SessionListResponse,
    SessionMode,
    SessionResponse,
    SessionStatus,
    SessionUpdate,
)


class TestSessionMode:
    """Tests for SessionMode enum."""

    def test_session_mode_values(self) -> None:
        """Test SessionMode enum has expected values."""
        assert SessionMode.TEXT.value == "text"
        assert SessionMode.VOICE.value == "voice"


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_session_status_values(self) -> None:
        """Test SessionStatus enum has expected values."""
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.ENDED.value == "ended"


class TestSessionCreate:
    """Tests for SessionCreate schema."""

    def test_valid_session_create(self) -> None:
        """Test creating a valid session."""
        data = SessionCreate()
        assert data.mode == SessionMode.TEXT
        assert data.livekit_room is None
        assert data.metadata == {}

    def test_session_create_voice_mode(self) -> None:
        """Test creating voice session."""
        data = SessionCreate(mode=SessionMode.VOICE, livekit_room="test-room")
        assert data.mode == SessionMode.VOICE
        assert data.livekit_room == "test-room"

    def test_session_create_with_metadata(self) -> None:
        """Test creating session with metadata."""
        data = SessionCreate(metadata={"key": "value"})
        assert data.metadata == {"key": "value"}

    def test_session_create_invalid_mode(self) -> None:
        """Test creating session with invalid mode raises error."""
        with pytest.raises(ValidationError):
            SessionCreate(mode="invalid")  # type: ignore[arg-type]


class TestSessionUpdate:
    """Tests for SessionUpdate schema."""

    def test_session_update_empty(self) -> None:
        """Test creating an empty update."""
        data = SessionUpdate()
        assert data.status is None
        assert data.livekit_room is None
        assert data.metadata is None

    def test_session_update_status(self) -> None:
        """Test updating status."""
        data = SessionUpdate(status=SessionStatus.ENDED)
        assert data.status == SessionStatus.ENDED

    def test_session_update_livekit_room(self) -> None:
        """Test updating livekit room."""
        data = SessionUpdate(livekit_room="new-room")
        assert data.livekit_room == "new-room"

    def test_session_update_metadata(self) -> None:
        """Test updating metadata."""
        data = SessionUpdate(metadata={"new": "value"})
        assert data.metadata == {"new": "value"}


class TestSessionResponse:
    """Tests for SessionResponse schema."""

    def test_session_response_from_dict(self) -> None:
        """Test creating response from dict."""
        session_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        now = datetime.now(UTC)

        data = SessionResponse(
            id=session_id,
            thread_id=thread_id,
            org_id=org_id,
            mode="text",
            status="active",
            livekit_room=None,
            metadata={"key": "value"},
            started_at=now,
            ended_at=None,
        )

        assert data.id == session_id
        assert data.thread_id == thread_id
        assert data.org_id == org_id
        assert data.mode == "text"
        assert data.status == "active"
        assert data.livekit_room is None
        assert data.metadata == {"key": "value"}

    def test_session_response_from_attributes(self) -> None:
        """Test that from_attributes is enabled."""
        assert SessionResponse.model_config.get("from_attributes") is True

    def test_session_response_from_orm_with_metadata(self) -> None:
        """Test from_orm_with_metadata class method."""
        session_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        now = datetime.now(UTC)

        # Create a mock ORM object
        mock_obj = MagicMock()
        mock_obj.id = session_id
        mock_obj.thread_id = thread_id
        mock_obj.org_id = org_id
        mock_obj.mode = "voice"
        mock_obj.status = "active"
        mock_obj.livekit_room = "test-room"
        mock_obj.metadata_ = {"key": "value"}
        mock_obj.started_at = now
        mock_obj.ended_at = None

        result = SessionResponse.from_orm_with_metadata(mock_obj)

        assert result.id == session_id
        assert result.mode == "voice"
        assert result.livekit_room == "test-room"
        assert result.metadata == {"key": "value"}

    def test_session_response_with_ended_at(self) -> None:
        """Test session response with ended_at timestamp."""
        session_id = uuid.uuid4()
        now = datetime.now(UTC)

        data = SessionResponse(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            mode="text",
            status="ended",
            livekit_room=None,
            metadata={},
            started_at=now,
            ended_at=now,
        )

        assert data.ended_at == now


class TestSessionListResponse:
    """Tests for SessionListResponse schema."""

    def test_session_list_response(self) -> None:
        """Test creating session list response."""
        session_id = uuid.uuid4()
        now = datetime.now(UTC)

        session = SessionResponse(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            mode="text",
            status="active",
            livekit_room=None,
            metadata={},
            started_at=now,
            ended_at=None,
        )

        data = SessionListResponse(sessions=[session], total=1)
        assert len(data.sessions) == 1
        assert data.total == 1

    def test_session_list_response_empty(self) -> None:
        """Test empty session list response."""
        data = SessionListResponse(sessions=[], total=0)
        assert len(data.sessions) == 0
        assert data.total == 0
