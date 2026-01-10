"""Unit tests for LiveKit API routes."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from priority_lens.api.routes.livekit import (
    router,
    set_livekit_service,
)
from priority_lens.services.livekit_service import (
    LiveKitNotConfiguredError,
    LiveKitService,
)


@pytest.fixture
def mock_user() -> MagicMock:
    """Create mock authenticated user."""
    user = MagicMock()
    user.id = "user_123"
    return user


@pytest.fixture
def mock_livekit_service() -> MagicMock:
    """Create mock LiveKit service."""
    service = MagicMock(spec=LiveKitService)
    service.is_configured = True
    service.get_server_url.return_value = "wss://test.livekit.cloud"
    service.get_room_name.return_value = "pl-thread-123"
    service.create_token.return_value = "mock-jwt-token"
    return service


@pytest.fixture
def mock_livekit_service_not_configured() -> MagicMock:
    """Create mock LiveKit service that is not configured."""
    service = MagicMock(spec=LiveKitService)
    service.is_configured = False
    service.get_server_url.return_value = None
    service.create_token.side_effect = LiveKitNotConfiguredError(
        "LiveKit API key and secret must be configured"
    )
    return service


@pytest.fixture
def app(mock_user: MagicMock, mock_livekit_service: MagicMock) -> FastAPI:
    """Create test FastAPI app with mocked dependencies."""
    from priority_lens.api.auth.dependencies import get_current_user_or_api_key

    app = FastAPI()
    app.include_router(router)

    # Override authentication dependency
    app.dependency_overrides[get_current_user_or_api_key] = lambda: mock_user

    # Set mock service
    set_livekit_service(mock_livekit_service)

    yield app

    # Cleanup
    set_livekit_service(None)
    app.dependency_overrides.clear()


@pytest.fixture
def app_not_configured(
    mock_user: MagicMock, mock_livekit_service_not_configured: MagicMock
) -> FastAPI:
    """Create test app with unconfigured LiveKit."""
    from priority_lens.api.auth.dependencies import get_current_user_or_api_key

    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_current_user_or_api_key] = lambda: mock_user
    set_livekit_service(mock_livekit_service_not_configured)

    yield app

    set_livekit_service(None)
    app.dependency_overrides.clear()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def client_not_configured(app_not_configured: FastAPI) -> TestClient:
    """Create test client with unconfigured LiveKit."""
    return TestClient(app_not_configured)


class TestGetConfig:
    """Tests for GET /livekit/config endpoint."""

    def test_config_enabled(self, client: TestClient, mock_livekit_service: MagicMock) -> None:
        """Test getting config when LiveKit is enabled."""
        response = client.get("/livekit/config")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["url"] == "wss://test.livekit.cloud"

    def test_config_disabled(self, client_not_configured: TestClient) -> None:
        """Test getting config when LiveKit is disabled."""
        response = client_not_configured.get("/livekit/config")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["url"] is None


class TestCreateToken:
    """Tests for POST /livekit/token endpoint."""

    def test_create_token_success(
        self, client: TestClient, mock_livekit_service: MagicMock
    ) -> None:
        """Test successful token creation."""
        thread_id = str(uuid4())
        session_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
                "participant_name": "Test User",
                "ttl_seconds": 300,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["token"] == "mock-jwt-token"
        assert data["room_name"] == "pl-thread-123"
        assert data["livekit_url"] == "wss://test.livekit.cloud"
        assert data["expires_in"] == 300

        # Verify service was called correctly
        mock_livekit_service.create_token.assert_called_once()

    def test_create_token_default_values(
        self, client: TestClient, mock_livekit_service: MagicMock
    ) -> None:
        """Test token creation with default values."""
        thread_id = str(uuid4())
        session_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["expires_in"] == 120  # Default TTL

    def test_create_token_not_configured(self, client_not_configured: TestClient) -> None:
        """Test token creation when LiveKit is not configured."""
        thread_id = str(uuid4())
        session_id = str(uuid4())

        response = client_not_configured.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
            },
        )

        assert response.status_code == 503
        assert "must be configured" in response.json()["detail"]

    def test_create_token_invalid_thread_id(self, client: TestClient) -> None:
        """Test token creation with invalid thread_id."""
        session_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "thread_id": "not-a-uuid",
                "session_id": session_id,
            },
        )

        assert response.status_code == 422
        assert "thread_id" in response.text

    def test_create_token_invalid_session_id(self, client: TestClient) -> None:
        """Test token creation with invalid session_id."""
        thread_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": "not-a-uuid",
            },
        )

        assert response.status_code == 422
        assert "session_id" in response.text

    def test_create_token_missing_thread_id(self, client: TestClient) -> None:
        """Test token creation with missing thread_id."""
        session_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "session_id": session_id,
            },
        )

        assert response.status_code == 422

    def test_create_token_ttl_too_low(self, client: TestClient) -> None:
        """Test token creation with TTL below minimum."""
        thread_id = str(uuid4())
        session_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
                "ttl_seconds": 10,  # Below 30 minimum
            },
        )

        assert response.status_code == 422

    def test_create_token_ttl_too_high(self, client: TestClient) -> None:
        """Test token creation with TTL above maximum."""
        thread_id = str(uuid4())
        session_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
                "ttl_seconds": 7200,  # Above 3600 maximum
            },
        )

        assert response.status_code == 422

    def test_create_token_no_server_url(
        self, client: TestClient, mock_livekit_service: MagicMock
    ) -> None:
        """Test token creation when server URL is not configured."""
        mock_livekit_service.get_server_url.return_value = None

        thread_id = str(uuid4())
        session_id = str(uuid4())

        response = client.post(
            "/livekit/token",
            json={
                "thread_id": thread_id,
                "session_id": session_id,
            },
        )

        assert response.status_code == 503
        assert "URL not configured" in response.json()["detail"]


class TestSetLiveKitService:
    """Tests for set_livekit_service function."""

    def test_set_and_reset_service(self) -> None:
        """Test setting and resetting the LiveKit service."""
        mock_service = MagicMock(spec=LiveKitService)

        # Set service
        set_livekit_service(mock_service)

        # Reset service
        set_livekit_service(None)
