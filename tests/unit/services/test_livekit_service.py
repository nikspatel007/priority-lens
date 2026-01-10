"""Unit tests for LiveKitService."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from priority_lens.api.config import APIConfig
from priority_lens.services.livekit_service import (
    LiveKitNotConfiguredError,
    LiveKitService,
)


@pytest.fixture
def config_with_livekit() -> APIConfig:
    """Create APIConfig with LiveKit credentials."""
    return APIConfig(
        livekit_api_key="test-api-key",
        livekit_api_secret="test-api-secret",
        livekit_url="wss://test.livekit.cloud",
    )


@pytest.fixture
def config_without_livekit() -> APIConfig:
    """Create APIConfig without LiveKit credentials."""
    return APIConfig(
        livekit_api_key=None,
        livekit_api_secret=None,
        livekit_url=None,
    )


def create_mock_livekit_module() -> ModuleType:
    """Create a mock livekit module with api submodule."""
    mock_token = MagicMock()
    mock_token.with_identity.return_value = mock_token
    mock_token.with_name.return_value = mock_token
    mock_token.with_ttl.return_value = mock_token
    mock_token.with_grants.return_value = mock_token
    mock_token.with_metadata.return_value = mock_token
    mock_token.to_jwt.return_value = "mock-jwt-token"

    mock_api = MagicMock()
    mock_api.AccessToken.return_value = mock_token
    mock_api.VideoGrants = MagicMock()

    mock_livekit = MagicMock()
    mock_livekit.api = mock_api

    return mock_livekit


class TestLiveKitService:
    """Tests for LiveKitService class."""

    def test_init_with_credentials(self, config_with_livekit: APIConfig) -> None:
        """Test service initialization with credentials."""
        service = LiveKitService(config_with_livekit)

        assert service.is_configured is True
        assert service._api_key == "test-api-key"
        assert service._api_secret == "test-api-secret"
        assert service._url == "wss://test.livekit.cloud"

    def test_init_without_credentials(self, config_without_livekit: APIConfig) -> None:
        """Test service initialization without credentials."""
        service = LiveKitService(config_without_livekit)

        assert service.is_configured is False
        assert service._api_key is None
        assert service._api_secret is None
        assert service._url is None

    def test_get_room_name(self, config_with_livekit: APIConfig) -> None:
        """Test room name generation."""
        service = LiveKitService(config_with_livekit)
        thread_id = UUID("12345678-1234-5678-1234-567812345678")

        room_name = service.get_room_name(thread_id)

        assert room_name == "pl-thread-12345678-1234-5678-1234-567812345678"

    def test_get_room_name_different_threads(self, config_with_livekit: APIConfig) -> None:
        """Test that different threads get different room names."""
        service = LiveKitService(config_with_livekit)
        thread_id_1 = uuid4()
        thread_id_2 = uuid4()

        room_name_1 = service.get_room_name(thread_id_1)
        room_name_2 = service.get_room_name(thread_id_2)

        assert room_name_1 != room_name_2
        assert room_name_1.startswith("pl-thread-")
        assert room_name_2.startswith("pl-thread-")

    def test_get_server_url_configured(self, config_with_livekit: APIConfig) -> None:
        """Test getting server URL when configured."""
        service = LiveKitService(config_with_livekit)

        url = service.get_server_url()

        assert url == "wss://test.livekit.cloud"

    def test_get_server_url_not_configured(self, config_without_livekit: APIConfig) -> None:
        """Test getting server URL when not configured."""
        service = LiveKitService(config_without_livekit)

        url = service.get_server_url()

        assert url is None

    def test_create_token_not_configured_raises(self, config_without_livekit: APIConfig) -> None:
        """Test that create_token raises when not configured."""
        service = LiveKitService(config_without_livekit)
        thread_id = uuid4()
        session_id = uuid4()

        with pytest.raises(LiveKitNotConfiguredError) as exc_info:
            service.create_token(
                thread_id=thread_id,
                session_id=session_id,
                participant_identity="user-123",
            )

        assert "must be configured" in str(exc_info.value)

    def test_create_token_import_error_raises(self, config_with_livekit: APIConfig) -> None:
        """Test that create_token raises when livekit-api is not installed."""
        service = LiveKitService(config_with_livekit)
        thread_id = uuid4()
        session_id = uuid4()

        # Save original livekit module if it exists
        original_livekit = sys.modules.get("livekit")

        try:
            # Remove livekit from sys.modules to simulate it not being installed
            if "livekit" in sys.modules:
                del sys.modules["livekit"]

            # Patch the import inside the function using builtins
            import builtins

            original_import = builtins.__import__

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "livekit":
                    raise ImportError("No module named 'livekit'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", mock_import):
                with pytest.raises(LiveKitNotConfiguredError) as exc_info:
                    service.create_token(
                        thread_id=thread_id,
                        session_id=session_id,
                        participant_identity="user-123",
                    )

            assert "livekit-api package is required" in str(exc_info.value)

        finally:
            # Restore original state
            if original_livekit is not None:
                sys.modules["livekit"] = original_livekit

    def test_create_token_success(self, config_with_livekit: APIConfig) -> None:
        """Test successful token creation with mocked livekit.api."""
        service = LiveKitService(config_with_livekit)
        thread_id = uuid4()
        session_id = uuid4()

        mock_livekit = create_mock_livekit_module()

        with patch.dict(sys.modules, {"livekit": mock_livekit}):
            token = service.create_token(
                thread_id=thread_id,
                session_id=session_id,
                participant_identity="user-123",
                participant_name="Test User",
                ttl_seconds=300,
            )

        assert token == "mock-jwt-token"

        # Verify the API was called correctly
        mock_livekit.api.AccessToken.assert_called_once_with("test-api-key", "test-api-secret")

    def test_create_token_default_values(self, config_with_livekit: APIConfig) -> None:
        """Test create_token with default parameter values."""
        service = LiveKitService(config_with_livekit)
        thread_id = uuid4()
        session_id = uuid4()

        mock_livekit = create_mock_livekit_module()

        with patch.dict(sys.modules, {"livekit": mock_livekit}):
            token = service.create_token(
                thread_id=thread_id,
                session_id=session_id,
                participant_identity="user-123",
            )

        assert token == "mock-jwt-token"

    def test_create_token_with_custom_name(self, config_with_livekit: APIConfig) -> None:
        """Test create_token with custom participant name."""
        service = LiveKitService(config_with_livekit)
        thread_id = uuid4()
        session_id = uuid4()

        mock_livekit = create_mock_livekit_module()

        with patch.dict(sys.modules, {"livekit": mock_livekit}):
            token = service.create_token(
                thread_id=thread_id,
                session_id=session_id,
                participant_identity="user-456",
                participant_name="Custom Name",
            )

        assert token == "mock-jwt-token"

        # Verify with_name was called
        mock_token = mock_livekit.api.AccessToken.return_value
        mock_token.with_name.assert_called()

    def test_create_token_with_custom_ttl(self, config_with_livekit: APIConfig) -> None:
        """Test create_token with custom TTL."""
        service = LiveKitService(config_with_livekit)
        thread_id = uuid4()
        session_id = uuid4()

        mock_livekit = create_mock_livekit_module()

        with patch.dict(sys.modules, {"livekit": mock_livekit}):
            token = service.create_token(
                thread_id=thread_id,
                session_id=session_id,
                participant_identity="user-789",
                ttl_seconds=600,
            )

        assert token == "mock-jwt-token"

        # Verify with_ttl was called
        mock_token = mock_livekit.api.AccessToken.return_value
        mock_token.with_ttl.assert_called()

    def test_create_token_calls_all_methods(self, config_with_livekit: APIConfig) -> None:
        """Test that create_token calls all required token builder methods."""
        service = LiveKitService(config_with_livekit)
        thread_id = uuid4()
        session_id = uuid4()

        mock_livekit = create_mock_livekit_module()

        with patch.dict(sys.modules, {"livekit": mock_livekit}):
            service.create_token(
                thread_id=thread_id,
                session_id=session_id,
                participant_identity="test-user",
            )

        mock_token = mock_livekit.api.AccessToken.return_value

        # Verify all builder methods were called
        mock_token.with_identity.assert_called_once_with("test-user")
        mock_token.with_name.assert_called_once()
        mock_token.with_ttl.assert_called_once()
        mock_token.with_grants.assert_called_once()
        mock_token.with_metadata.assert_called_once()
        mock_token.to_jwt.assert_called_once()


class TestLiveKitNotConfiguredError:
    """Tests for LiveKitNotConfiguredError exception."""

    def test_exception_message(self) -> None:
        """Test exception contains message."""
        error = LiveKitNotConfiguredError("Test error message")
        assert str(error) == "Test error message"

    def test_exception_is_exception_subclass(self) -> None:
        """Test exception inherits from Exception."""
        error = LiveKitNotConfiguredError("test")
        assert isinstance(error, Exception)
