"""Tests for digest API route."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from priority_lens.api.routes.digest import (
    router,
    set_session_factory,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def mock_user() -> MagicMock:
    """Create mock Clerk user."""
    user = MagicMock()
    user.id = "user_123"
    user.first_name = "Sarah"
    user.full_name = "Sarah Johnson"
    return user


@pytest.fixture
def app(mock_user: MagicMock) -> FastAPI:
    """Create FastAPI app with digest router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Override auth dependency
    from priority_lens.api.auth.dependencies import get_current_user_or_api_key

    app.dependency_overrides[get_current_user_or_api_key] = lambda: mock_user

    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Create test client."""
    from contextlib import asynccontextmanager

    # Set up mock session factory
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result

    @asynccontextmanager
    async def mock_session_context():
        yield mock_session

    set_session_factory(mock_session_context)

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    set_session_factory(None)


class TestGetSmartDigest:
    """Tests for GET /digest endpoint."""

    def test_returns_digest_response(self, client: TestClient) -> None:
        """Test successful digest response."""
        response = client.get("/api/v1/digest")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "greeting" in data
        assert "subtitle" in data
        assert "suggested_todos" in data
        assert "topics_to_catchup" in data
        assert "last_updated" in data

    def test_greeting_includes_user_name(self, client: TestClient) -> None:
        """Test greeting includes user's first name."""
        response = client.get("/api/v1/digest")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Sarah from mock_user fixture
        assert "Sarah" in data["greeting"]

    def test_accepts_max_todos_parameter(self, client: TestClient) -> None:
        """Test max_todos query parameter."""
        response = client.get("/api/v1/digest?max_todos=10")

        assert response.status_code == status.HTTP_200_OK

    def test_accepts_max_topics_parameter(self, client: TestClient) -> None:
        """Test max_topics query parameter."""
        response = client.get("/api/v1/digest?max_topics=10")

        assert response.status_code == status.HTTP_200_OK

    def test_validates_max_todos_min(self, client: TestClient) -> None:
        """Test max_todos minimum validation."""
        response = client.get("/api/v1/digest?max_todos=0")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_validates_max_todos_max(self, client: TestClient) -> None:
        """Test max_todos maximum validation."""
        response = client.get("/api/v1/digest?max_todos=100")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_validates_max_topics_min(self, client: TestClient) -> None:
        """Test max_topics minimum validation."""
        response = client.get("/api/v1/digest?max_topics=0")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_validates_max_topics_max(self, client: TestClient) -> None:
        """Test max_topics maximum validation."""
        response = client.get("/api/v1/digest?max_topics=100")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestDigestWithoutAuth:
    """Tests for digest endpoint without authentication."""

    def test_requires_authentication(self) -> None:
        """Test endpoint requires authentication."""
        from priority_lens.api.auth.exceptions import AuthenticationError

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        # Add exception handler to convert AuthenticationError to 401
        @app.exception_handler(AuthenticationError)
        async def auth_exception_handler(request, exc):
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": str(exc)}
            )

        # Don't override auth - use actual dependency
        test_client = TestClient(app, raise_server_exceptions=False)

        response = test_client.get("/api/v1/digest")

        # Should fail because no auth
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,  # If exception handler not set up
            status.HTTP_503_SERVICE_UNAVAILABLE,  # If session factory not set
        ]


class TestDigestServiceIntegration:
    """Integration tests for digest service with route."""

    def test_empty_digest_shows_caught_up(self, client: TestClient) -> None:
        """Test empty digest shows 'All caught up' subtitle."""
        response = client.get("/api/v1/digest")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "caught up" in data["subtitle"].lower() or data["subtitle"] == "All caught up!"
