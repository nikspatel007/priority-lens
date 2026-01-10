"""Tests for session repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from priority_lens.models.session import Session
from priority_lens.repositories.session import SessionRepository
from priority_lens.schemas.session import SessionCreate, SessionMode, SessionStatus, SessionUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> SessionRepository:
    """Create repository with mock session."""
    return SessionRepository(mock_session)


class TestSessionRepository:
    """Tests for SessionRepository."""

    @pytest.mark.asyncio
    async def test_create_session(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a session."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        data = SessionCreate(mode=SessionMode.TEXT, metadata={"key": "value"})

        result = await repository.create(data, thread_id, org_id)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, Session)
        assert result.thread_id == thread_id
        assert result.org_id == org_id
        assert result.mode == "text"
        assert result.metadata_ == {"key": "value"}

    @pytest.mark.asyncio
    async def test_create_voice_session(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a voice session."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        data = SessionCreate(mode=SessionMode.VOICE, livekit_room="test-room")

        result = await repository.create(data, thread_id, org_id)

        assert result.mode == "voice"
        assert result.livekit_room == "test-room"

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting session by ID when found."""
        session_id = uuid.uuid4()
        session = Session(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            mode="text",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = session
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(session_id)

        assert result == session

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting session by ID when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_and_org_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting session by ID and org when found."""
        session_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session = Session(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=org_id,
            mode="text",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = session
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id_and_org(session_id, org_id)

        assert result == session

    @pytest.mark.asyncio
    async def test_list_by_thread(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing sessions for a thread."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        sessions = [
            Session(id=uuid.uuid4(), thread_id=thread_id, org_id=org_id, mode="text"),
            Session(id=uuid.uuid4(), thread_id=thread_id, org_id=org_id, mode="voice"),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sessions
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_thread(thread_id, org_id)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_count_by_thread(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test counting sessions for a thread."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 3
        mock_session.execute.return_value = mock_result

        result = await repository.count_by_thread(thread_id, org_id)

        assert result == 3

    @pytest.mark.asyncio
    async def test_get_active_session_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting active session when found."""
        thread_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session = Session(
            id=uuid.uuid4(),
            thread_id=thread_id,
            org_id=org_id,
            mode="text",
            status="active",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = session
        mock_session.execute.return_value = mock_result

        result = await repository.get_active_session(thread_id, org_id)

        assert result == session
        assert result.status == "active"

    @pytest.mark.asyncio
    async def test_get_active_session_not_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting active session when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_active_session(uuid.uuid4(), uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_update_status(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating session status."""
        session_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session = Session(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=org_id,
            mode="text",
            status="active",
        )
        data = SessionUpdate(status=SessionStatus.ENDED)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = session
        mock_session.execute.return_value = mock_result

        result = await repository.update(session_id, org_id, data)

        assert result is not None
        assert result.status == "ended"

    @pytest.mark.asyncio
    async def test_update_livekit_room(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating session livekit room."""
        session_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session = Session(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=org_id,
            mode="voice",
            livekit_room=None,
        )
        data = SessionUpdate(livekit_room="new-room")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = session
        mock_session.execute.return_value = mock_result

        result = await repository.update(session_id, org_id, data)

        assert result is not None
        assert result.livekit_room == "new-room"

    @pytest.mark.asyncio
    async def test_update_metadata(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating session metadata."""
        session_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session = Session(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=org_id,
            mode="text",
            metadata_={},
        )
        data = SessionUpdate(metadata={"new": "value"})

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = session
        mock_session.execute.return_value = mock_result

        result = await repository.update(session_id, org_id, data)

        assert result is not None
        assert result.metadata_ == {"new": "value"}

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating session when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.update(
            uuid.uuid4(), uuid.uuid4(), SessionUpdate(status=SessionStatus.ENDED)
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_end_session_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test ending session when found."""
        session_id = uuid.uuid4()
        org_id = uuid.uuid4()
        session = Session(
            id=session_id,
            thread_id=uuid.uuid4(),
            org_id=org_id,
            mode="text",
            status="active",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = session
        mock_session.execute.return_value = mock_result

        with patch("priority_lens.repositories.session.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = mock_now

            result = await repository.end_session(session_id, org_id)

            assert result is not None
            assert result.status == "ended"
            assert result.ended_at == mock_now

    @pytest.mark.asyncio
    async def test_end_session_not_found(
        self, repository: SessionRepository, mock_session: AsyncMock
    ) -> None:
        """Test ending session when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.end_session(uuid.uuid4(), uuid.uuid4())

        assert result is None
