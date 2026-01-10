"""Integration tests for Voice AI and SDUI flow.

These tests verify the full conversation flow from thread creation
through event streaming and action handling.

Run with: uv run pytest tests/integration/test_voice_sdui_flow.py -v
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from priority_lens.api.config import APIConfig
from priority_lens.api.routes import (
    set_actions_session,
    set_agent_session,
    set_threads_session,
)
from priority_lens.api.routes.livekit import set_livekit_service
from priority_lens.models.canonical_event import EventType
from priority_lens.services.livekit_service import LiveKitService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def get_test_db_url() -> str:
    """Get test database URL."""
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5433/test_priority_lens",
    )


def is_database_available() -> bool:
    """Check if the test database is available."""
    import asyncio

    async def check() -> bool:
        try:
            engine = create_async_engine(get_test_db_url(), echo=False)
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            return True
        except Exception:
            return False

    try:
        return asyncio.get_event_loop().run_until_complete(check())
    except RuntimeError:
        return asyncio.run(check())


# Skip all tests in this module if database is not available
pytestmark = pytest.mark.skipif(not is_database_available(), reason="Test database not available")


@pytest.fixture(scope="module")
def engine():
    """Create async engine for tests."""
    return create_async_engine(get_test_db_url(), echo=False)


@pytest.fixture(scope="module")
def session_maker(engine):
    """Create session maker."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(session_maker) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for tests with rollback."""
    async with session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    user = MagicMock()
    user.id = str(uuid4())
    user.email = "test@example.com"
    user.first_name = "Test"
    user.last_name = "User"
    return user


@pytest.fixture
def mock_livekit_service():
    """Create a mock LiveKit service."""
    service = MagicMock(spec=LiveKitService)
    service.is_configured = True
    service.get_room_name.return_value = "pl-thread-test-room"
    service.create_token.return_value = "mock-livekit-token"
    service.get_server_url.return_value = "wss://test.livekit.cloud"
    return service


@pytest.fixture
def mock_config():
    """Create a mock API config."""
    config = MagicMock(spec=APIConfig)
    config.environment = "test"
    config.clerk_secret_key = "test-secret"
    config.has_livekit = True
    config.livekit_api_key = "test-api-key"
    config.livekit_api_secret = "test-api-secret"
    config.livekit_url = "wss://test.livekit.cloud"
    return config


@pytest.fixture
async def test_app(session_maker, mock_user, mock_livekit_service, mock_config):
    """Create test FastAPI app with mocked dependencies."""
    from priority_lens.api.routes import (
        actions_router,
        agent_router,
        livekit_router,
        threads_router,
    )

    app = FastAPI()
    app.include_router(threads_router)
    app.include_router(livekit_router)
    app.include_router(agent_router)
    app.include_router(actions_router)

    @asynccontextmanager
    async def session_factory():
        async with session_maker() as session:
            yield session

    set_threads_session(session_factory)
    set_agent_session(session_factory)
    set_actions_session(session_factory)
    set_livekit_service(mock_livekit_service)

    # Override auth dependency
    from priority_lens.api.auth.dependencies import get_current_user_or_api_key

    app.dependency_overrides[get_current_user_or_api_key] = lambda: mock_user

    yield app

    # Cleanup
    set_threads_session(None)
    set_agent_session(None)
    set_actions_session(None)
    set_livekit_service(None)
    app.dependency_overrides.clear()


@pytest.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestThreadLifecycle:
    """Integration tests for thread lifecycle."""

    @pytest.mark.asyncio
    async def test_create_thread(self, client: AsyncClient) -> None:
        """Test creating a new thread."""
        response = await client.post(
            "/threads",
            json={"title": "Test Thread", "metadata": {"source": "integration_test"}},
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["title"] == "Test Thread"
        assert data["metadata"]["source"] == "integration_test"

    @pytest.mark.asyncio
    async def test_list_threads(self, client: AsyncClient) -> None:
        """Test listing threads."""
        # Create a thread first
        create_response = await client.post("/threads", json={"title": "List Test"})
        assert create_response.status_code == 201

        # List threads
        response = await client.get("/threads")
        assert response.status_code == 200
        data = response.json()
        assert "threads" in data
        assert "total" in data
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_get_thread(self, client: AsyncClient) -> None:
        """Test getting a specific thread."""
        # Create a thread first
        create_response = await client.post("/threads", json={"title": "Get Test"})
        assert create_response.status_code == 201
        thread_id = create_response.json()["id"]

        # Get thread
        response = await client.get(f"/threads/{thread_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == thread_id
        assert data["title"] == "Get Test"

    @pytest.mark.asyncio
    async def test_get_thread_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent thread."""
        fake_id = str(uuid4())
        response = await client.get(f"/threads/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_thread(self, client: AsyncClient) -> None:
        """Test updating a thread."""
        # Create a thread first
        create_response = await client.post("/threads", json={"title": "Original Title"})
        assert create_response.status_code == 201
        thread_id = create_response.json()["id"]

        # Update thread
        response = await client.patch(
            f"/threads/{thread_id}",
            json={"title": "Updated Title"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"

    @pytest.mark.asyncio
    async def test_delete_thread(self, client: AsyncClient) -> None:
        """Test deleting a thread."""
        # Create a thread first
        create_response = await client.post("/threads", json={"title": "Delete Test"})
        assert create_response.status_code == 201
        thread_id = create_response.json()["id"]

        # Delete thread
        response = await client.delete(f"/threads/{thread_id}")
        assert response.status_code == 204

        # Verify deleted
        get_response = await client.get(f"/threads/{thread_id}")
        assert get_response.status_code == 404


class TestSessionLifecycle:
    """Integration tests for session lifecycle."""

    @pytest.mark.asyncio
    async def test_create_text_session(self, client: AsyncClient) -> None:
        """Test creating a text session."""
        # Create a thread first
        thread_response = await client.post("/threads", json={"title": "Session Test"})
        assert thread_response.status_code == 201
        thread_id = thread_response.json()["id"]

        # Create session
        response = await client.post(
            f"/threads/{thread_id}/sessions",
            json={"mode": "text", "metadata": {"device": "web"}},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["thread_id"] == thread_id
        assert data["mode"] == "text"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_create_voice_session(self, client: AsyncClient) -> None:
        """Test creating a voice session."""
        # Create a thread first
        thread_response = await client.post("/threads", json={"title": "Voice Session Test"})
        assert thread_response.status_code == 201
        thread_id = thread_response.json()["id"]

        # Create session
        response = await client.post(
            f"/threads/{thread_id}/sessions",
            json={"mode": "voice", "livekit_room": "test-room"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["mode"] == "voice"
        assert data["livekit_room"] == "test-room"

    @pytest.mark.asyncio
    async def test_list_sessions(self, client: AsyncClient) -> None:
        """Test listing sessions for a thread."""
        # Create a thread and session
        thread_response = await client.post("/threads", json={"title": "List Sessions Test"})
        thread_id = thread_response.json()["id"]

        await client.post(f"/threads/{thread_id}/sessions", json={"mode": "text"})
        await client.post(f"/threads/{thread_id}/sessions", json={"mode": "voice"})

        # List sessions
        response = await client.get(f"/threads/{thread_id}/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["sessions"]) == 2


class TestTurnSubmission:
    """Integration tests for turn submission."""

    @pytest.mark.asyncio
    async def test_submit_text_turn(self, client: AsyncClient) -> None:
        """Test submitting a text turn."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Turn Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit turn
        response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Show my priority inbox"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["accepted"] is True
        assert data["thread_id"] == thread_id
        assert data["session_id"] == session_id
        assert "correlation_id" in data
        assert "seq" in data

    @pytest.mark.asyncio
    async def test_submit_voice_turn(self, client: AsyncClient) -> None:
        """Test submitting a voice turn."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Voice Turn Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "voice"}
        )
        session_id = session_response.json()["id"]

        # Submit turn
        response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {
                    "type": "voice",
                    "transcript": "What are my tasks for today",
                    "confidence": 0.95,
                    "duration_ms": 2500,
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["accepted"] is True


class TestEventPersistence:
    """Integration tests for event persistence."""

    @pytest.mark.asyncio
    async def test_events_created_on_turn(self, client: AsyncClient) -> None:
        """Test that events are created when submitting a turn."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Events Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit turn
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Test message"},
            },
        )

        # Get events
        response = await client.get(f"/threads/{thread_id}/events")
        assert response.status_code == 200
        data = response.json()

        # Should have at least 3 events: turn.user.open, ui.text.submit, turn.user.close
        assert len(data["events"]) >= 3

        # Verify event types
        event_types = [e["type"] for e in data["events"]]
        assert EventType.TURN_USER_OPEN.value in event_types
        assert EventType.UI_TEXT_SUBMIT.value in event_types
        assert EventType.TURN_USER_CLOSE.value in event_types

    @pytest.mark.asyncio
    async def test_events_monotonic_sequence(self, client: AsyncClient) -> None:
        """Test that events have monotonically increasing sequence numbers."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Sequence Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit multiple turns
        for i in range(3):
            await client.post(
                f"/threads/{thread_id}/turns",
                json={
                    "session_id": session_id,
                    "input": {"type": "text", "text": f"Message {i}"},
                },
            )

        # Get events
        response = await client.get(f"/threads/{thread_id}/events")
        data = response.json()

        # Verify sequence is monotonically increasing
        seqs = [e["seq"] for e in data["events"]]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], "Sequence numbers must be monotonically increasing"

    @pytest.mark.asyncio
    async def test_events_after_seq(self, client: AsyncClient) -> None:
        """Test fetching events after a specific sequence number."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "After Seq Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit first turn
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "First message"},
            },
        )

        # Get first batch of events
        response1 = await client.get(f"/threads/{thread_id}/events")
        first_batch = response1.json()
        first_seq = first_batch["next_seq"]

        # Submit second turn
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Second message"},
            },
        )

        # Get events after first_seq
        response2 = await client.get(f"/threads/{thread_id}/events?after_seq={first_seq}")
        second_batch = response2.json()

        # Second batch should only have events from the second turn
        for event in second_batch["events"]:
            assert event["seq"] > first_seq


class TestLiveKitIntegration:
    """Integration tests for LiveKit token generation."""

    @pytest.mark.asyncio
    async def test_get_livekit_config(self, client: AsyncClient) -> None:
        """Test getting LiveKit configuration."""
        response = await client.get("/livekit/config")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "url" in data

    @pytest.mark.asyncio
    async def test_create_livekit_token(self, client: AsyncClient) -> None:
        """Test creating a LiveKit token."""
        # Create thread and session first
        thread_response = await client.post("/threads", json={"title": "LiveKit Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "voice"}
        )
        session_id = session_response.json()["id"]

        # Create token
        response = await client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
                "participant_name": "Test User",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "room_name" in data
        assert "livekit_url" in data
        assert "expires_in" in data


class TestActionHandling:
    """Integration tests for action handling."""

    @pytest.mark.asyncio
    async def test_list_action_types(self, client: AsyncClient) -> None:
        """Test listing available action types."""
        response = await client.get("/actions/types")
        assert response.status_code == 200
        data = response.json()
        assert "types" in data
        assert "archive" in data["types"]
        assert "complete" in data["types"]
        assert "snooze" in data["types"]

    @pytest.mark.asyncio
    async def test_execute_archive_action(self, client: AsyncClient) -> None:
        """Test executing an archive action."""
        # Create thread for context
        thread_response = await client.post("/threads", json={"title": "Action Test"})
        thread_id = thread_response.json()["id"]

        response = await client.post(
            "/actions",
            json={
                "id": "action-test-123",
                "type": "archive",
                "thread_id": thread_id,
                "payload": {"email_id": 456},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_navigate_action(self, client: AsyncClient) -> None:
        """Test executing a navigate action."""
        thread_response = await client.post("/threads", json={"title": "Navigate Test"})
        thread_id = thread_response.json()["id"]

        response = await client.post(
            "/actions",
            json={
                "id": "action-nav-123",
                "type": "navigate",
                "thread_id": thread_id,
                "payload": {"route": "/projects/123"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_action_missing_payload(self, client: AsyncClient) -> None:
        """Test action with missing required payload."""
        thread_response = await client.post("/threads", json={"title": "Missing Payload Test"})
        thread_id = thread_response.json()["id"]

        response = await client.post(
            "/actions",
            json={
                "id": "action-missing-123",
                "type": "archive",
                "thread_id": thread_id,
                "payload": {},  # Missing email_id
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is False
        assert data["status"] == "failure"
        assert data["error"] == "MISSING_PARAM"


class TestAgentControl:
    """Integration tests for agent control."""

    @pytest.mark.asyncio
    async def test_cancel_agent(self, client: AsyncClient) -> None:
        """Test cancelling agent execution."""
        correlation_id = str(uuid4())

        response = await client.post(
            "/agent/cancel",
            json={"correlation_id": correlation_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert "cancelled" in data
        assert data["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_get_agent_status(self, client: AsyncClient) -> None:
        """Test getting agent status."""
        correlation_id = str(uuid4())

        response = await client.get(f"/agent/status/{correlation_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["correlation_id"] == correlation_id
        assert "active" in data


class TestFullConversationFlow:
    """Integration tests for full conversation flow."""

    @pytest.mark.asyncio
    async def test_complete_text_conversation_flow(self, client: AsyncClient) -> None:
        """Test a complete text conversation flow."""
        # 1. Create thread
        thread_response = await client.post(
            "/threads",
            json={"title": "Full Flow Test", "metadata": {"source": "integration"}},
        )
        assert thread_response.status_code == 201
        thread_id = thread_response.json()["id"]

        # 2. Create session
        session_response = await client.post(
            f"/threads/{thread_id}/sessions",
            json={"mode": "text"},
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["id"]

        # 3. Submit turn
        turn_response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Show my priority inbox"},
            },
        )
        assert turn_response.status_code == 201
        correlation_id = turn_response.json()["correlation_id"]

        # 4. Verify events
        events_response = await client.get(f"/threads/{thread_id}/events")
        assert events_response.status_code == 200
        events = events_response.json()["events"]

        # Verify turn events
        event_types = [e["type"] for e in events]
        assert EventType.TURN_USER_OPEN.value in event_types
        assert EventType.UI_TEXT_SUBMIT.value in event_types
        assert EventType.TURN_USER_CLOSE.value in event_types

        # Verify correlation IDs match
        turn_events = [e for e in events if e["payload"].get("correlation_id")]
        for event in turn_events:
            assert event["payload"]["correlation_id"] == str(correlation_id)

        # 5. Execute action (simulating response to SDUI)
        action_response = await client.post(
            "/actions",
            json={
                "id": "action-complete-flow",
                "type": "archive",
                "thread_id": thread_id,
                "payload": {"email_id": 123},
            },
        )
        assert action_response.status_code == 200
        assert action_response.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_reconnection_flow(self, client: AsyncClient) -> None:
        """Test reconnection using after_seq."""
        # 1. Create thread and session
        thread_response = await client.post("/threads", json={"title": "Reconnect Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # 2. Submit first turn
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "First message"},
            },
        )

        # 3. Get current events (simulating initial connection)
        initial_events = await client.get(f"/threads/{thread_id}/events")
        last_seq = initial_events.json()["next_seq"]

        # 4. Submit more turns (simulating activity while disconnected)
        for i in range(3):
            await client.post(
                f"/threads/{thread_id}/turns",
                json={
                    "session_id": session_id,
                    "input": {"type": "text", "text": f"Message while disconnected {i}"},
                },
            )

        # 5. Reconnect using after_seq
        reconnect_events = await client.get(f"/threads/{thread_id}/events?after_seq={last_seq}")
        data = reconnect_events.json()

        # Should have events from the 3 turns submitted while "disconnected"
        # Each turn creates 3 events: open, content, close = 9 events
        assert len(data["events"]) >= 9

        # All events should be after last_seq
        for event in data["events"]:
            assert event["seq"] > last_seq
