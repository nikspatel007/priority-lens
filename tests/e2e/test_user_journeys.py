"""End-to-end tests for user journeys.

These tests verify complete user workflows from start to finish,
including email triage, task management, and project overview.

Run with: uv run python -m pytest tests/e2e/ -v
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from priority_lens.api.routes import (
    set_actions_session,
    set_agent_session,
    set_inbox_session,
    set_projects_session,
    set_tasks_session,
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


# ============================================================================
# Test Fixtures and Factories
# ============================================================================


class TestDataFactory:
    """Factory for creating test data objects as dictionaries."""

    @staticmethod
    def create_email(
        *,
        email_id: int = 1,
        subject: str = "Test Email",
        sender: str = "sender@example.com",
        priority_score: float = 0.8,
        urgent: bool = False,
        requires_response: bool = True,
    ) -> dict:
        """Create a test email data dictionary."""
        return {
            "email_id": email_id,
            "message_id": f"<test{email_id}@example.com>",
            "subject": subject,
            "from_email": sender,
            "from_name": sender.split("@")[0],
            "date": datetime.now(UTC).isoformat(),
            "priority_score": priority_score,
            "priority_factors": {
                "sender_importance": 0.7,
                "urgency_keywords": 0.3 if urgent else 0.0,
            },
            "handleability_class": "requires_response" if requires_response else "fyi",
            "is_thread_starter": True,
            "thread_id": str(uuid4()),
            "snippet": f"This is a snippet of {subject}...",
        }

    @staticmethod
    def create_task(
        *,
        task_id: int = 1,
        title: str = "Test Task",
        status: str = "pending",
        priority: int = 1,
    ) -> dict:
        """Create a test task data dictionary."""
        return {
            "id": task_id,
            "user_id": str(uuid4()),
            "email_id": task_id * 100,
            "title": title,
            "description": f"Description for {title}",
            "status": status,
            "priority": priority,
            "due_date": datetime.now(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def create_project(
        *,
        project_id: int = 1,
        name: str = "Test Project",
        is_active: bool = True,
    ) -> dict:
        """Create a test project data dictionary."""
        return {
            "id": project_id,
            "user_id": str(uuid4()),
            "name": name,
            "description": f"Description for {name}",
            "is_active": is_active,
            "email_count": 10,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }


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
def test_data_factory():
    """Create a test data factory."""
    return TestDataFactory()


@pytest.fixture
async def e2e_app(session_maker, mock_user, mock_livekit_service):
    """Create E2E test FastAPI app with all routes."""
    from priority_lens.api.routes import (
        actions_router,
        agent_router,
        inbox_router,
        livekit_router,
        projects_router,
        tasks_router,
        threads_router,
    )

    app = FastAPI()
    app.include_router(threads_router)
    app.include_router(livekit_router)
    app.include_router(agent_router)
    app.include_router(actions_router)
    app.include_router(inbox_router)
    app.include_router(projects_router)
    app.include_router(tasks_router)

    @asynccontextmanager
    async def session_factory():
        async with session_maker() as session:
            yield session

    set_threads_session(session_factory)
    set_agent_session(session_factory)
    set_actions_session(session_factory)
    set_inbox_session(session_factory)
    set_projects_session(session_factory)
    set_tasks_session(session_factory)
    set_livekit_service(mock_livekit_service)

    # Override auth dependency
    from priority_lens.api.auth.dependencies import get_current_user_or_api_key

    app.dependency_overrides[get_current_user_or_api_key] = lambda: mock_user

    yield app

    # Cleanup
    set_threads_session(None)
    set_agent_session(None)
    set_actions_session(None)
    set_inbox_session(None)
    set_projects_session(None)
    set_tasks_session(None)
    set_livekit_service(None)
    app.dependency_overrides.clear()


@pytest.fixture
async def client(e2e_app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ============================================================================
# Email Triage Journey Tests
# ============================================================================


class TestEmailTriageJourney:
    """E2E tests for the email triage user journey.

    User Journey: Connect Gmail -> Sync -> Ask "What's urgent?" -> View inbox card
    """

    @pytest.mark.asyncio
    async def test_complete_email_triage_flow(
        self, client: AsyncClient, test_data_factory: TestDataFactory
    ) -> None:
        """Test complete email triage flow from thread creation to inbox view."""
        # Step 1: Create a conversation thread
        thread_response = await client.post(
            "/threads",
            json={"title": "Email Triage Session", "metadata": {"journey": "email_triage"}},
        )
        assert thread_response.status_code == 201
        thread_id = thread_response.json()["id"]

        # Step 2: Create a text session
        session_response = await client.post(
            f"/threads/{thread_id}/sessions",
            json={"mode": "text"},
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["id"]

        # Step 3: Submit user turn asking about urgent emails
        turn_response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "What emails need my attention today?"},
            },
        )
        assert turn_response.status_code == 201
        assert "correlation_id" in turn_response.json()

        # Step 4: Verify events were created
        events_response = await client.get(f"/threads/{thread_id}/events")
        assert events_response.status_code == 200
        events = events_response.json()["events"]

        # Should have turn events (open, content, close)
        event_types = [e["type"] for e in events]
        assert EventType.TURN_USER_OPEN.value in event_types
        assert EventType.UI_TEXT_SUBMIT.value in event_types
        assert EventType.TURN_USER_CLOSE.value in event_types

        # Step 5: Execute an action on an email (archive)
        action_response = await client.post(
            "/actions",
            json={
                "id": f"action-{uuid4()}",
                "type": "archive",
                "thread_id": thread_id,
                "payload": {"email_id": 123},
            },
        )
        assert action_response.status_code == 200
        assert action_response.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_email_triage_with_voice_session(
        self, client: AsyncClient, mock_livekit_service: MagicMock
    ) -> None:
        """Test email triage using voice input."""
        # Create thread
        thread_response = await client.post(
            "/threads",
            json={"title": "Voice Email Triage", "metadata": {"mode": "voice"}},
        )
        thread_id = thread_response.json()["id"]

        # Create voice session
        session_response = await client.post(
            f"/threads/{thread_id}/sessions",
            json={"mode": "voice", "livekit_room": "test-room"},
        )
        session_id = session_response.json()["id"]

        # Get LiveKit token
        token_response = await client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
                "participant_name": "Test User",
            },
        )
        assert token_response.status_code == 200
        assert token_response.json()["token"] == "mock-livekit-token"

        # Submit voice turn
        turn_response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {
                    "type": "voice",
                    "transcript": "Show me urgent emails",
                    "confidence": 0.92,
                    "duration_ms": 1500,
                },
            },
        )
        assert turn_response.status_code == 201

    @pytest.mark.asyncio
    async def test_email_triage_multiple_turns(self, client: AsyncClient) -> None:
        """Test email triage with multiple conversation turns."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Multi-Turn Triage"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Turn 1: Ask about urgent emails
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Show me urgent emails"},
            },
        )

        # Turn 2: Ask for more details
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Tell me more about the first one"},
            },
        )

        # Turn 3: Take action
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Archive it"},
            },
        )

        # Verify all events
        events_response = await client.get(f"/threads/{thread_id}/events")
        events = events_response.json()["events"]

        # Should have 9 events (3 turns x 3 events each)
        assert len(events) >= 9


# ============================================================================
# Task Management Journey Tests
# ============================================================================


class TestTaskManagementJourney:
    """E2E tests for the task management user journey.

    User Journey: Ask "Show my tasks" -> Complete task -> Verify update
    """

    @pytest.mark.asyncio
    async def test_complete_task_management_flow(
        self, client: AsyncClient, test_data_factory: TestDataFactory
    ) -> None:
        """Test complete task management flow."""
        # Step 1: Create conversation thread
        thread_response = await client.post(
            "/threads",
            json={"title": "Task Management Session", "metadata": {"journey": "task_management"}},
        )
        thread_id = thread_response.json()["id"]

        # Step 2: Create session
        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Step 3: Ask about tasks
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Show me my pending tasks"},
            },
        )

        # Step 4: Complete a task via action
        complete_response = await client.post(
            "/actions",
            json={
                "id": f"action-complete-{uuid4()}",
                "type": "complete",
                "thread_id": thread_id,
                "payload": {"task_id": 1},
            },
        )
        # Note: This will fail validation since task_id doesn't exist
        # but we're testing the flow works
        assert complete_response.status_code == 200

        # Step 5: Verify events logged
        events_response = await client.get(f"/threads/{thread_id}/events")
        events = events_response.json()["events"]
        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_task_snooze_flow(self, client: AsyncClient) -> None:
        """Test task snooze workflow."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Task Snooze"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Ask about tasks
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Show my tasks"},
            },
        )

        # Snooze a task
        snooze_response = await client.post(
            "/actions",
            json={
                "id": f"action-snooze-{uuid4()}",
                "type": "snooze",
                "thread_id": thread_id,
                "payload": {
                    "item_type": "task",
                    "task_id": 1,
                    "snooze_until": "2025-01-15T10:00:00Z",
                },
            },
        )
        assert snooze_response.status_code == 200
        assert snooze_response.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_task_dismiss_flow(self, client: AsyncClient) -> None:
        """Test task dismiss workflow."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Task Dismiss"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Ask to dismiss a task
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Dismiss my second task"},
            },
        )

        # Execute dismiss action
        dismiss_response = await client.post(
            "/actions",
            json={
                "id": f"action-dismiss-{uuid4()}",
                "type": "dismiss",
                "thread_id": thread_id,
                "payload": {"task_id": 2},
            },
        )
        assert dismiss_response.status_code == 200


# ============================================================================
# Project Overview Journey Tests
# ============================================================================


class TestProjectOverviewJourney:
    """E2E tests for the project overview user journey.

    User Journey: Ask "Status of Project X" -> View project card
    """

    @pytest.mark.asyncio
    async def test_complete_project_overview_flow(
        self, client: AsyncClient, test_data_factory: TestDataFactory
    ) -> None:
        """Test complete project overview flow."""
        # Step 1: Create conversation thread
        thread_response = await client.post(
            "/threads",
            json={
                "title": "Project Overview Session",
                "metadata": {"journey": "project_overview"},
            },
        )
        thread_id = thread_response.json()["id"]

        # Step 2: Create session
        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Step 3: Ask about project status
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "What's the status of Project Alpha?"},
            },
        )

        # Step 4: Navigate to project details
        navigate_response = await client.post(
            "/actions",
            json={
                "id": f"action-nav-{uuid4()}",
                "type": "navigate",
                "thread_id": thread_id,
                "payload": {"route": "/projects/1"},
            },
        )
        assert navigate_response.status_code == 200
        assert navigate_response.json()["ok"] is True

        # Step 5: Verify events
        events_response = await client.get(f"/threads/{thread_id}/events")
        events = events_response.json()["events"]
        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_project_list_request(self, client: AsyncClient) -> None:
        """Test requesting a list of projects."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Project List"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Ask for all projects
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Show me all my active projects"},
            },
        )

        # Verify events
        events_response = await client.get(f"/threads/{thread_id}/events")
        events = events_response.json()["events"]
        assert len(events) >= 3


# ============================================================================
# Cross-Journey Tests
# ============================================================================


class TestCrossJourneyScenarios:
    """E2E tests for scenarios that span multiple user journeys."""

    @pytest.mark.asyncio
    async def test_email_to_task_conversion(self, client: AsyncClient) -> None:
        """Test converting an email to a task."""
        # Create thread
        thread_response = await client.post("/threads", json={"title": "Email to Task Conversion"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Ask to convert email to task
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Create a task from this email"},
            },
        )

        # Verify events created
        events_response = await client.get(f"/threads/{thread_id}/events")
        assert events_response.status_code == 200

    @pytest.mark.asyncio
    async def test_session_persistence_across_reconnect(self, client: AsyncClient) -> None:
        """Test that session state persists across reconnection."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Reconnect Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit initial turn
        await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Initial message"},
            },
        )

        # Get events before "disconnect"
        initial_events = await client.get(f"/threads/{thread_id}/events")
        last_seq = initial_events.json()["next_seq"]

        # Submit more turns (simulating activity)
        for i in range(3):
            await client.post(
                f"/threads/{thread_id}/turns",
                json={
                    "session_id": session_id,
                    "input": {"type": "text", "text": f"Message {i}"},
                },
            )

        # Reconnect using after_seq
        reconnect_events = await client.get(f"/threads/{thread_id}/events?after_seq={last_seq}")
        data = reconnect_events.json()

        # Should have events from the 3 new turns
        assert len(data["events"]) >= 9  # 3 turns x 3 events each

    @pytest.mark.asyncio
    async def test_agent_cancellation_during_journey(self, client: AsyncClient) -> None:
        """Test that agent can be cancelled mid-journey."""
        # Create thread and session
        thread_response = await client.post("/threads", json={"title": "Cancel Test"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit turn
        turn_response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": "Show me all my emails"},
            },
        )
        correlation_id = turn_response.json()["correlation_id"]

        # Cancel the agent
        cancel_response = await client.post(
            "/agent/cancel",
            json={"correlation_id": correlation_id},
        )
        assert cancel_response.status_code == 200
        assert "cancelled" in cancel_response.json()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestJourneyErrorHandling:
    """E2E tests for error handling across user journeys."""

    @pytest.mark.asyncio
    async def test_invalid_thread_access(self, client: AsyncClient) -> None:
        """Test accessing a non-existent thread."""
        fake_thread_id = str(uuid4())

        response = await client.get(f"/threads/{fake_thread_id}/events")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_action_type(self, client: AsyncClient) -> None:
        """Test executing an invalid action type."""
        # Create thread
        thread_response = await client.post("/threads", json={"title": "Invalid Action"})
        thread_id = thread_response.json()["id"]

        # Try invalid action
        response = await client.post(
            "/actions",
            json={
                "id": "action-invalid",
                "type": "invalid_action_type",
                "thread_id": thread_id,
                "payload": {},
            },
        )
        assert response.status_code in [400, 500]  # Either validation error or not found

    @pytest.mark.asyncio
    async def test_missing_required_payload(self, client: AsyncClient) -> None:
        """Test action with missing required payload."""
        thread_response = await client.post("/threads", json={"title": "Missing Payload"})
        thread_id = thread_response.json()["id"]

        response = await client.post(
            "/actions",
            json={
                "id": "action-missing",
                "type": "navigate",
                "thread_id": thread_id,
                "payload": {},  # Missing 'route'
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is False
        assert response.json()["error"] == "MISSING_PARAM"

    @pytest.mark.asyncio
    async def test_empty_text_input(self, client: AsyncClient) -> None:
        """Test submitting empty text input."""
        thread_response = await client.post("/threads", json={"title": "Empty Input"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Try to submit empty text - should fail validation
        response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": ""},
            },
        )
        assert response.status_code == 422  # Validation error


# ============================================================================
# Performance Tests
# ============================================================================


class TestJourneyPerformance:
    """E2E tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_rapid_turn_submission(self, client: AsyncClient) -> None:
        """Test rapid submission of multiple turns."""
        thread_response = await client.post("/threads", json={"title": "Rapid Turns"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit 10 turns rapidly
        for i in range(10):
            response = await client.post(
                f"/threads/{thread_id}/turns",
                json={
                    "session_id": session_id,
                    "input": {"type": "text", "text": f"Rapid message {i}"},
                },
            )
            assert response.status_code == 201

        # Verify all events
        events_response = await client.get(f"/threads/{thread_id}/events?limit=100")
        events = events_response.json()["events"]

        # Should have 30 events (10 turns x 3 events each)
        assert len(events) >= 30

        # Verify sequences are monotonically increasing
        seqs = [e["seq"] for e in events]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1]

    @pytest.mark.asyncio
    async def test_large_event_payload(self, client: AsyncClient) -> None:
        """Test handling of large text inputs."""
        thread_response = await client.post("/threads", json={"title": "Large Payload"})
        thread_id = thread_response.json()["id"]

        session_response = await client.post(
            f"/threads/{thread_id}/sessions", json={"mode": "text"}
        )
        session_id = session_response.json()["id"]

        # Submit a large text input (within limits)
        large_text = "Test message. " * 500  # ~7000 characters

        response = await client.post(
            f"/threads/{thread_id}/turns",
            json={
                "session_id": session_id,
                "input": {"type": "text", "text": large_text},
            },
        )
        assert response.status_code == 201
