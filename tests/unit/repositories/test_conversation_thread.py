"""Tests for conversation thread repository."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from priority_lens.models.conversation_thread import ConversationThread
from priority_lens.repositories.conversation_thread import ThreadRepository
from priority_lens.schemas.conversation_thread import ThreadCreate, ThreadUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> ThreadRepository:
    """Create repository with mock session."""
    return ThreadRepository(mock_session)


class TestThreadRepository:
    """Tests for ThreadRepository."""

    @pytest.mark.asyncio
    async def test_create_thread(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a thread."""
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        data = ThreadCreate(title="Test Thread", metadata={"key": "value"})

        result = await repository.create(data, org_id, user_id)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, ConversationThread)
        assert result.title == "Test Thread"
        assert result.org_id == org_id
        assert result.user_id == user_id
        assert result.metadata_ == {"key": "value"}

    @pytest.mark.asyncio
    async def test_create_thread_null_title(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a thread with null title."""
        org_id = uuid.uuid4()
        user_id = uuid.uuid4()
        data = ThreadCreate()

        result = await repository.create(data, org_id, user_id)

        assert result.title is None
        assert result.metadata_ == {}

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting thread by ID when found."""
        thread_id = uuid.uuid4()
        thread = ConversationThread(
            id=thread_id,
            org_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            title="Test",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = thread
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(thread_id)

        assert result == thread
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting thread by ID when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_and_org_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting thread by ID and org when found."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=uuid.uuid4(),
            title="Test",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = thread
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id_and_org(thread_id, org_id)

        assert result == thread

    @pytest.mark.asyncio
    async def test_get_by_id_and_org_not_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting thread by ID and org when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id_and_org(uuid.uuid4(), uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_user(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing threads for a user."""
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        threads = [
            ConversationThread(id=uuid.uuid4(), org_id=org_id, user_id=user_id, title="Thread 1"),
            ConversationThread(id=uuid.uuid4(), org_id=org_id, user_id=user_id, title="Thread 2"),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = threads
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_user(user_id, org_id)

        assert len(result) == 2
        assert result == threads

    @pytest.mark.asyncio
    async def test_list_by_user_with_pagination(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing threads with pagination."""
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        threads = [ConversationThread(id=uuid.uuid4(), org_id=org_id, user_id=user_id)]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = threads
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_user(user_id, org_id, limit=10, offset=5)

        assert len(result) == 1
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_by_user(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test counting threads for a user."""
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 5
        mock_session.execute.return_value = mock_result

        result = await repository.count_by_user(user_id, org_id)

        assert result == 5

    @pytest.mark.asyncio
    async def test_update_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating thread when found."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=uuid.uuid4(),
            title="Old Title",
        )
        data = ThreadUpdate(title="New Title")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = thread
        mock_session.execute.return_value = mock_result

        result = await repository.update(thread_id, org_id, data)

        assert result is not None
        assert result.title == "New Title"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_metadata(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating thread metadata."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=uuid.uuid4(),
            title="Test",
            metadata_={},
        )
        data = ThreadUpdate(metadata={"new": "value"})

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = thread
        mock_session.execute.return_value = mock_result

        result = await repository.update(thread_id, org_id, data)

        assert result is not None
        assert result.metadata_ == {"new": "value"}

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating thread when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.update(uuid.uuid4(), uuid.uuid4(), ThreadUpdate(title="New"))

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting thread when found."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=uuid.uuid4(),
            title="Test",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = thread
        mock_session.execute.return_value = mock_result

        result = await repository.delete(thread_id, org_id)

        assert result is True
        mock_session.delete.assert_called_once_with(thread)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, repository: ThreadRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting thread when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.delete(uuid.uuid4(), uuid.uuid4())

        assert result is False
        mock_session.delete.assert_not_called()
