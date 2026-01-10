"""Tests for conversation thread schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from priority_lens.schemas.conversation_thread import (
    ThreadCreate,
    ThreadListResponse,
    ThreadResponse,
    ThreadUpdate,
)


class TestThreadCreate:
    """Tests for ThreadCreate schema."""

    def test_valid_thread_create(self) -> None:
        """Test creating a valid thread."""
        data = ThreadCreate(title="Test Thread")
        assert data.title == "Test Thread"
        assert data.metadata == {}

    def test_thread_create_with_metadata(self) -> None:
        """Test creating thread with metadata."""
        data = ThreadCreate(
            title="Test Thread",
            metadata={"key": "value"},
        )
        assert data.metadata == {"key": "value"}

    def test_thread_create_null_title(self) -> None:
        """Test creating thread with null title."""
        data = ThreadCreate()
        assert data.title is None
        assert data.metadata == {}

    def test_thread_create_empty_metadata(self) -> None:
        """Test creating thread with empty metadata."""
        data = ThreadCreate(title="Test", metadata={})
        assert data.metadata == {}


class TestThreadUpdate:
    """Tests for ThreadUpdate schema."""

    def test_thread_update_empty(self) -> None:
        """Test creating an empty update."""
        data = ThreadUpdate()
        assert data.title is None
        assert data.metadata is None

    def test_thread_update_title_only(self) -> None:
        """Test updating title only."""
        data = ThreadUpdate(title="New Title")
        assert data.title == "New Title"
        assert data.metadata is None

    def test_thread_update_metadata_only(self) -> None:
        """Test updating metadata only."""
        data = ThreadUpdate(metadata={"new": "value"})
        assert data.title is None
        assert data.metadata == {"new": "value"}

    def test_thread_update_both_fields(self) -> None:
        """Test updating both fields."""
        data = ThreadUpdate(title="New Title", metadata={"key": "value"})
        assert data.title == "New Title"
        assert data.metadata == {"key": "value"}


class TestThreadResponse:
    """Tests for ThreadResponse schema."""

    def test_thread_response_from_dict(self) -> None:
        """Test creating response from dict."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now(UTC)

        data = ThreadResponse(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title="Test Thread",
            metadata={"key": "value"},
            created_at=now,
            updated_at=now,
        )

        assert data.id == thread_id
        assert data.org_id == org_id
        assert data.user_id == user_id
        assert data.title == "Test Thread"
        assert data.metadata == {"key": "value"}

    def test_thread_response_from_attributes(self) -> None:
        """Test that from_attributes is enabled."""
        assert ThreadResponse.model_config.get("from_attributes") is True

    def test_thread_response_from_orm_with_metadata(self) -> None:
        """Test from_orm_with_metadata class method."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now(UTC)

        # Create a mock ORM object
        mock_obj = MagicMock()
        mock_obj.id = thread_id
        mock_obj.org_id = org_id
        mock_obj.user_id = user_id
        mock_obj.title = "Test Thread"
        mock_obj.metadata_ = {"key": "value"}
        mock_obj.created_at = now
        mock_obj.updated_at = now

        result = ThreadResponse.from_orm_with_metadata(mock_obj)

        assert result.id == thread_id
        assert result.org_id == org_id
        assert result.user_id == user_id
        assert result.title == "Test Thread"
        assert result.metadata == {"key": "value"}


class TestThreadListResponse:
    """Tests for ThreadListResponse schema."""

    def test_thread_list_response(self) -> None:
        """Test creating thread list response."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now(UTC)

        thread = ThreadResponse(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title="Test Thread",
            metadata={},
            created_at=now,
            updated_at=now,
        )

        data = ThreadListResponse(threads=[thread], total=1)
        assert len(data.threads) == 1
        assert data.total == 1

    def test_thread_list_response_empty(self) -> None:
        """Test empty thread list response."""
        data = ThreadListResponse(threads=[], total=0)
        assert len(data.threads) == 0
        assert data.total == 0
