"""Tests for agent API routes."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from priority_lens.api.routes.agent import (
    AgentStatusResponse,
    CancelRequest,
    CancelResponse,
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


@pytest.fixture
def mock_user() -> MagicMock:
    """Create mock authenticated user."""
    user = MagicMock()
    user.id = str(uuid4())
    return user


class TestCancelRequest:
    """Tests for CancelRequest schema."""

    def test_cancel_request_creation(self) -> None:
        """Test creating a cancel request."""
        correlation_id = uuid4()
        request = CancelRequest(correlation_id=correlation_id)

        assert request.correlation_id == correlation_id
        assert request.reason == "user_request"

    def test_cancel_request_with_custom_reason(self) -> None:
        """Test cancel request with custom reason."""
        correlation_id = uuid4()
        request = CancelRequest(
            correlation_id=correlation_id,
            reason="timeout",
        )

        assert request.reason == "timeout"


class TestCancelResponse:
    """Tests for CancelResponse schema."""

    def test_cancel_response_success(self) -> None:
        """Test creating a successful cancel response."""
        correlation_id = uuid4()
        response = CancelResponse(
            ok=True,
            correlation_id=correlation_id,
            message="Agent execution cancelled",
        )

        assert response.ok is True
        assert response.correlation_id == correlation_id
        assert response.message == "Agent execution cancelled"

    def test_cancel_response_failure(self) -> None:
        """Test creating a failed cancel response."""
        correlation_id = uuid4()
        response = CancelResponse(
            ok=False,
            correlation_id=correlation_id,
            message="No active session found",
        )

        assert response.ok is False


class TestAgentStatusResponse:
    """Tests for AgentStatusResponse schema."""

    def test_agent_status_response(self) -> None:
        """Test creating an agent status response."""
        correlation_id = uuid4()
        response = AgentStatusResponse(
            active_sessions=5,
            is_session_active=True,
            correlation_id=correlation_id,
        )

        assert response.active_sessions == 5
        assert response.is_session_active is True
        assert response.correlation_id == correlation_id

    def test_agent_status_response_no_correlation(self) -> None:
        """Test agent status response without correlation ID."""
        response = AgentStatusResponse(
            active_sessions=0,
            is_session_active=False,
        )

        assert response.correlation_id is None


class TestCancelAgentEndpoint:
    """Tests for POST /agent/cancel endpoint."""

    def test_cancel_request_validation(self) -> None:
        """Test cancel request validation."""
        # Valid request
        request = CancelRequest(
            correlation_id=uuid4(),
            reason="barge_in",
        )
        assert request.reason == "barge_in"

        # Request with default reason
        request2 = CancelRequest(correlation_id=uuid4())
        assert request2.reason == "user_request"

    def test_cancel_request_serialization(self) -> None:
        """Test cancel request JSON serialization."""
        correlation_id = uuid4()
        request = CancelRequest(
            correlation_id=correlation_id,
            reason="timeout",
        )

        data = request.model_dump()
        assert data["correlation_id"] == correlation_id
        assert data["reason"] == "timeout"


class TestAgentStatusEndpoint:
    """Tests for GET /agent/status/{correlation_id} endpoint."""

    def test_agent_status_schema(self) -> None:
        """Test agent status response schema."""
        correlation_id = uuid4()

        response = AgentStatusResponse(
            active_sessions=3,
            is_session_active=True,
            correlation_id=correlation_id,
        )

        assert response.model_dump() == {
            "active_sessions": 3,
            "is_session_active": True,
            "correlation_id": correlation_id,
        }


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_prefix(self) -> None:
        """Test that router has correct prefix."""
        assert router.prefix == "/agent"

    def test_router_tags(self) -> None:
        """Test that router has correct tags."""
        assert "agent" in router.tags

    def test_routes_registered(self) -> None:
        """Test that expected routes are registered."""
        route_paths = [route.path for route in router.routes]

        # Routes include the prefix
        assert "/agent/cancel" in route_paths
        assert "/agent/status/{correlation_id}" in route_paths
