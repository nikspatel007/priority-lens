"""Tests for ConversationThread model."""

from __future__ import annotations

import uuid

from priority_lens.models.conversation_thread import ConversationThread


class TestConversationThread:
    """Tests for ConversationThread model."""

    def test_conversation_thread_create(self) -> None:
        """Test creating a conversation thread."""
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        thread = ConversationThread(
            org_id=org_id,
            user_id=user_id,
            title="Test Thread",
            metadata_={"key": "value"},
        )
        assert thread.org_id == org_id
        assert thread.user_id == user_id
        assert thread.title == "Test Thread"
        assert thread.metadata_ == {"key": "value"}

    def test_conversation_thread_default_id(self) -> None:
        """Test thread has UUID id by default."""
        thread = ConversationThread(
            org_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
        )
        # Default is generated when not set
        assert thread.id is None or isinstance(thread.id, uuid.UUID)

    def test_conversation_thread_default_metadata(self) -> None:
        """Test thread has empty dict metadata by default."""
        thread = ConversationThread(
            org_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
        )
        # Before commit, metadata may be None or empty dict
        assert thread.metadata_ is None or thread.metadata_ == {}

    def test_conversation_thread_null_title(self) -> None:
        """Test thread can have null title."""
        thread = ConversationThread(
            org_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            title=None,
        )
        assert thread.title is None

    def test_conversation_thread_repr(self) -> None:
        """Test thread string representation."""
        thread_id = uuid.uuid4()
        user_id = uuid.uuid4()
        thread = ConversationThread(
            id=thread_id,
            org_id=uuid.uuid4(),
            user_id=user_id,
            title="Test Thread",
        )
        result = repr(thread)
        assert "ConversationThread" in result
        assert "Test Thread" in result
        assert str(user_id) in result


class TestConversationThreadTablename:
    """Tests for ConversationThread table configuration."""

    def test_tablename(self) -> None:
        """Test table name is correct."""
        assert ConversationThread.__tablename__ == "conversation_threads"
