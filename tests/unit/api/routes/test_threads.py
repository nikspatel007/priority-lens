"""Tests for thread API endpoints."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from priority_lens.api.auth.clerk import ClerkUser
from priority_lens.api.auth.dependencies import get_current_user_or_api_key
from priority_lens.api.routes.threads import router, set_session_factory
from priority_lens.models.canonical_event import CanonicalEvent
from priority_lens.models.conversation_thread import ConversationThread
from priority_lens.models.session import Session


@pytest.fixture
def mock_user() -> ClerkUser:
    """Create a mock authenticated user."""
    return ClerkUser(
        id="550e8400-e29b-41d4-a716-446655440000",  # Valid UUID format
        email="test@example.com",
        first_name="Test",
        last_name="User",
    )


@pytest.fixture
def mock_session() -> mock.MagicMock:
    """Create a mock database session."""
    return mock.MagicMock()


@pytest.fixture
def app(mock_user: ClerkUser, mock_session: mock.MagicMock) -> FastAPI:
    """Create a test FastAPI app."""
    from contextlib import asynccontextmanager

    app = FastAPI()
    app.include_router(router)

    async def override_get_current_user() -> ClerkUser:
        return mock_user

    app.dependency_overrides[get_current_user_or_api_key] = override_get_current_user

    @asynccontextmanager
    async def mock_session_cm() -> AsyncGenerator[mock.MagicMock, None]:
        yield mock_session

    set_session_factory(mock_session_cm)  # type: ignore[arg-type]

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestCreateThread:
    """Tests for create_thread endpoint."""

    def test_create_thread(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test creating a thread."""
        thread_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title="Test Thread",
            metadata_={"key": "value"},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.create = mock.AsyncMock(return_value=mock_thread)

            response = client.post(
                "/threads",
                json={"title": "Test Thread", "metadata": {"key": "value"}},
            )

            assert response.status_code == 201
            data = response.json()
            assert data["title"] == "Test Thread"
            assert data["metadata"] == {"key": "value"}

    def test_create_thread_null_title(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test creating a thread with null title."""
        thread_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title=None,
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.create = mock.AsyncMock(return_value=mock_thread)

            response = client.post("/threads", json={})

            assert response.status_code == 201
            data = response.json()
            assert data["title"] is None


class TestListThreads:
    """Tests for list_threads endpoint."""

    def test_list_threads(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test listing threads."""
        thread_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title="Test Thread",
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.list_by_user = mock.AsyncMock(return_value=[mock_thread])
            mock_repo.count_by_user = mock.AsyncMock(return_value=1)

            response = client.get("/threads")

            assert response.status_code == 200
            data = response.json()
            assert len(data["threads"]) == 1
            assert data["total"] == 1

    def test_list_threads_with_pagination(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test listing threads with pagination."""
        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.list_by_user = mock.AsyncMock(return_value=[])
            mock_repo.count_by_user = mock.AsyncMock(return_value=0)

            response = client.get("/threads?limit=10&offset=5")

            assert response.status_code == 200
            mock_repo.list_by_user.assert_called_once()


class TestGetThread:
    """Tests for get_thread endpoint."""

    def test_get_thread_success(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting a thread."""
        thread_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title="Test Thread",
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_by_id_and_org = mock.AsyncMock(return_value=mock_thread)

            response = client.get(f"/threads/{thread_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Test Thread"

    def test_get_thread_not_found(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting a non-existent thread."""
        thread_id = uuid.uuid4()

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_by_id_and_org = mock.AsyncMock(return_value=None)

            response = client.get(f"/threads/{thread_id}")

            assert response.status_code == 404
            assert str(thread_id) in response.json()["detail"]


class TestUpdateThread:
    """Tests for update_thread endpoint."""

    def test_update_thread(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test updating a thread."""
        thread_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title="Updated Thread",
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.update = mock.AsyncMock(return_value=mock_thread)

            response = client.patch(
                f"/threads/{thread_id}",
                json={"title": "Updated Thread"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Updated Thread"

    def test_update_thread_not_found(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test updating a non-existent thread."""
        thread_id = uuid.uuid4()

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.update = mock.AsyncMock(return_value=None)

            response = client.patch(
                f"/threads/{thread_id}",
                json={"title": "Updated"},
            )

            assert response.status_code == 404


class TestDeleteThread:
    """Tests for delete_thread endpoint."""

    def test_delete_thread(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test deleting a thread."""
        thread_id = uuid.uuid4()

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.delete = mock.AsyncMock(return_value=True)

            response = client.delete(f"/threads/{thread_id}")

            assert response.status_code == 204

    def test_delete_thread_not_found(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test deleting a non-existent thread."""
        thread_id = uuid.uuid4()

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.delete = mock.AsyncMock(return_value=False)

            response = client.delete(f"/threads/{thread_id}")

            assert response.status_code == 404


class TestGetEvents:
    """Tests for get_events endpoint."""

    def test_get_events(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting events."""
        thread_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=user_id,
            title="Test Thread",
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        mock_event = CanonicalEvent(
            event_id=uuid.uuid4(),
            thread_id=thread_id,
            org_id=org_id,
            seq=1,
            ts=int(now.timestamp() * 1000),
            actor="user",
            type="ui.text.submit",
            payload={"text": "Hello"},
        )

        with (
            mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockThreadRepo,
            mock.patch("priority_lens.api.routes.threads.EventRepository") as MockEventRepo,
        ):
            mock_thread_repo = MockThreadRepo.return_value
            mock_thread_repo.get_by_id_and_org = mock.AsyncMock(return_value=mock_thread)

            mock_event_repo = MockEventRepo.return_value
            mock_event_repo.get_events_after_seq = mock.AsyncMock(return_value=[mock_event])

            response = client.get(f"/threads/{thread_id}/events")

            assert response.status_code == 200
            data = response.json()
            assert len(data["events"]) == 1
            assert data["next_seq"] == 1
            assert data["has_more"] is False

    def test_get_events_with_after_seq(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test getting events after a sequence number."""
        thread_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=org_id,
            title="Test Thread",
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        with (
            mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockThreadRepo,
            mock.patch("priority_lens.api.routes.threads.EventRepository") as MockEventRepo,
        ):
            mock_thread_repo = MockThreadRepo.return_value
            mock_thread_repo.get_by_id_and_org = mock.AsyncMock(return_value=mock_thread)

            mock_event_repo = MockEventRepo.return_value
            mock_event_repo.get_events_after_seq = mock.AsyncMock(return_value=[])

            response = client.get(f"/threads/{thread_id}/events?after_seq=10")

            assert response.status_code == 200
            data = response.json()
            assert len(data["events"]) == 0
            assert data["next_seq"] == 10

    def test_get_events_thread_not_found(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test getting events from a non-existent thread."""
        thread_id = uuid.uuid4()

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_by_id_and_org = mock.AsyncMock(return_value=None)

            response = client.get(f"/threads/{thread_id}/events")

            assert response.status_code == 404


class TestCreateSession:
    """Tests for create_session endpoint."""

    def test_create_session(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test creating a session."""
        thread_id = uuid.uuid4()
        session_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=org_id,
            title="Test Thread",
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        mock_sess = Session(
            id=session_id,
            thread_id=thread_id,
            org_id=org_id,
            mode="text",
            status="active",
            metadata_={},
        )
        mock_sess.started_at = now

        with (
            mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockThreadRepo,
            mock.patch("priority_lens.api.routes.threads.SessionRepository") as MockSessionRepo,
        ):
            mock_thread_repo = MockThreadRepo.return_value
            mock_thread_repo.get_by_id_and_org = mock.AsyncMock(return_value=mock_thread)

            mock_session_repo = MockSessionRepo.return_value
            mock_session_repo.create = mock.AsyncMock(return_value=mock_sess)

            response = client.post(
                f"/threads/{thread_id}/sessions",
                json={"mode": "text"},
            )

            assert response.status_code == 201
            data = response.json()
            assert data["mode"] == "text"
            assert data["status"] == "active"

    def test_create_session_thread_not_found(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test creating a session on a non-existent thread."""
        thread_id = uuid.uuid4()

        with mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_by_id_and_org = mock.AsyncMock(return_value=None)

            response = client.post(
                f"/threads/{thread_id}/sessions",
                json={"mode": "text"},
            )

            assert response.status_code == 404


class TestListSessions:
    """Tests for list_sessions endpoint."""

    def test_list_sessions(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test listing sessions."""
        thread_id = uuid.uuid4()
        session_id = uuid.uuid4()
        org_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(UTC)

        mock_thread = ConversationThread(
            id=thread_id,
            org_id=org_id,
            user_id=org_id,
            title="Test Thread",
            metadata_={},
        )
        mock_thread.created_at = now
        mock_thread.updated_at = now

        mock_sess = Session(
            id=session_id,
            thread_id=thread_id,
            org_id=org_id,
            mode="text",
            status="active",
            metadata_={},
        )
        mock_sess.started_at = now

        with (
            mock.patch("priority_lens.api.routes.threads.ThreadRepository") as MockThreadRepo,
            mock.patch("priority_lens.api.routes.threads.SessionRepository") as MockSessionRepo,
        ):
            mock_thread_repo = MockThreadRepo.return_value
            mock_thread_repo.get_by_id_and_org = mock.AsyncMock(return_value=mock_thread)

            mock_session_repo = MockSessionRepo.return_value
            mock_session_repo.list_by_thread = mock.AsyncMock(return_value=[mock_sess])
            mock_session_repo.count_by_thread = mock.AsyncMock(return_value=1)

            response = client.get(f"/threads/{thread_id}/sessions")

            assert response.status_code == 200
            data = response.json()
            assert len(data["sessions"]) == 1
            assert data["total"] == 1


class TestDatabaseNotConfigured:
    """Tests for database not configured scenario."""

    def test_service_not_configured(self) -> None:
        """Test error when database not configured."""
        app = FastAPI()
        app.include_router(router)

        async def override_get_current_user() -> ClerkUser:
            return ClerkUser(id="550e8400-e29b-41d4-a716-446655440000")

        app.dependency_overrides[get_current_user_or_api_key] = override_get_current_user

        # Reset the session factory
        set_session_factory(None)

        client = TestClient(app)
        response = client.get("/threads")

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
