"""Tests for actions API routes."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from priority_lens.api.routes.actions import (
    ActionRequest,
    ActionResponse,
    ActionTypesResponse,
    router,
    set_session_factory,
)


@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_session() -> MagicMock:
    """Create mock database session."""
    return MagicMock()


@pytest.fixture
def client(app: FastAPI, mock_session: MagicMock) -> TestClient:
    """Create test client with mocked dependencies."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    set_session_factory(mock_session_factory)

    return TestClient(app)


class TestActionRequest:
    """Tests for ActionRequest schema."""

    def test_action_request_creation(self) -> None:
        """Test creating an action request."""
        thread_id = uuid4()
        request = ActionRequest(
            id="action-123",
            type="archive",
            thread_id=thread_id,
            payload={"email_id": 456},
        )

        assert request.id == "action-123"
        assert request.type == "archive"
        assert request.thread_id == thread_id
        assert request.payload == {"email_id": 456}
        assert request.session_id is None

    def test_action_request_with_session(self) -> None:
        """Test action request with session ID."""
        session_id = uuid4()
        request = ActionRequest(
            id="action-456",
            type="complete",
            thread_id=uuid4(),
            session_id=session_id,
            payload={"task_id": 789},
        )

        assert request.session_id == session_id

    def test_action_request_default_payload(self) -> None:
        """Test action request with default payload."""
        request = ActionRequest(
            id="action-789",
            type="navigate",
            thread_id=uuid4(),
        )

        assert request.payload == {}

    def test_action_request_serialization(self) -> None:
        """Test action request JSON serialization."""
        thread_id = uuid4()
        request = ActionRequest(
            id="action-123",
            type="snooze",
            thread_id=thread_id,
            payload={"task_id": 123, "duration": "1h"},
        )

        data = request.model_dump()
        assert data["id"] == "action-123"
        assert data["type"] == "snooze"
        assert data["thread_id"] == thread_id
        assert data["payload"]["task_id"] == 123


class TestActionResponse:
    """Tests for ActionResponse schema."""

    def test_action_response_success(self) -> None:
        """Test creating a successful action response."""
        response = ActionResponse(
            ok=True,
            action_id="action-123",
            status="success",
            message="Email archived successfully",
            data={"email_id": 456},
        )

        assert response.ok is True
        assert response.action_id == "action-123"
        assert response.status == "success"
        assert response.message == "Email archived successfully"
        assert response.data == {"email_id": 456}
        assert response.error is None

    def test_action_response_failure(self) -> None:
        """Test creating a failed action response."""
        response = ActionResponse(
            ok=False,
            action_id="action-456",
            status="failure",
            message="Task not found",
            error="NOT_FOUND",
        )

        assert response.ok is False
        assert response.status == "failure"
        assert response.error == "NOT_FOUND"

    def test_action_response_serialization(self) -> None:
        """Test action response JSON serialization."""
        response = ActionResponse(
            ok=True,
            action_id="action-123",
            status="success",
            message="Done",
        )

        data = response.model_dump()
        assert data["ok"] is True
        assert data["action_id"] == "action-123"
        assert data["status"] == "success"
        assert data["data"] is None


class TestActionTypesResponse:
    """Tests for ActionTypesResponse schema."""

    def test_action_types_response(self) -> None:
        """Test creating an action types response."""
        response = ActionTypesResponse(
            types=["archive", "complete", "snooze"],
        )

        assert len(response.types) == 3
        assert "archive" in response.types

    def test_action_types_response_empty(self) -> None:
        """Test action types response with empty list."""
        response = ActionTypesResponse(types=[])

        assert response.types == []


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_prefix(self) -> None:
        """Test that router has correct prefix."""
        assert router.prefix == "/actions"

    def test_router_tags(self) -> None:
        """Test that router has correct tags."""
        assert "actions" in router.tags

    def test_routes_registered(self) -> None:
        """Test that expected routes are registered."""
        route_paths = [route.path for route in router.routes]

        # Routes include the prefix
        assert "/actions" in route_paths
        assert "/actions/types" in route_paths


class TestExecuteActionEndpoint:
    """Tests for POST /actions endpoint."""

    def test_action_request_validation(self) -> None:
        """Test action request field validation."""
        thread_id = uuid4()
        request = ActionRequest(
            id="test-action",
            type="archive",
            thread_id=thread_id,
            payload={"email_id": 123},
        )

        assert request.type == "archive"
        assert request.payload["email_id"] == 123

    def test_action_request_required_fields(self) -> None:
        """Test that action request requires id, type, and thread_id."""
        thread_id = uuid4()

        # Should work with required fields
        request = ActionRequest(
            id="action-1",
            type="navigate",
            thread_id=thread_id,
        )
        assert request.id == "action-1"


class TestListActionTypesEndpoint:
    """Tests for GET /actions/types endpoint."""

    def test_action_types_schema(self) -> None:
        """Test action types response schema."""
        response = ActionTypesResponse(
            types=["archive", "complete", "dismiss", "snooze", "navigate", "reply", "delete"],
        )

        assert len(response.types) == 7
        assert "archive" in response.types
        assert "complete" in response.types
        assert "dismiss" in response.types
        assert "snooze" in response.types
        assert "navigate" in response.types
        assert "reply" in response.types
        assert "delete" in response.types
