"""Unit tests for LiveKit schemas."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from priority_lens.schemas.livekit import (
    LiveKitConfig,
    LiveKitTokenRequest,
    LiveKitTokenResponse,
)


class TestLiveKitTokenRequest:
    """Tests for LiveKitTokenRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid request with all fields."""
        thread_id = uuid4()
        session_id = uuid4()

        request = LiveKitTokenRequest(
            thread_id=thread_id,
            session_id=session_id,
            participant_name="Test User",
            ttl_seconds=300,
        )

        assert request.thread_id == thread_id
        assert request.session_id == session_id
        assert request.participant_name == "Test User"
        assert request.ttl_seconds == 300

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        thread_id = uuid4()
        session_id = uuid4()

        request = LiveKitTokenRequest(
            thread_id=thread_id,
            session_id=session_id,
        )

        assert request.participant_name == "user"
        assert request.ttl_seconds == 120

    def test_missing_thread_id_raises(self) -> None:
        """Test that missing thread_id raises validation error."""
        session_id = uuid4()

        with pytest.raises(ValidationError) as exc_info:
            LiveKitTokenRequest(session_id=session_id)  # type: ignore[call-arg]

        assert "thread_id" in str(exc_info.value)

    def test_missing_session_id_raises(self) -> None:
        """Test that missing session_id raises validation error."""
        thread_id = uuid4()

        with pytest.raises(ValidationError) as exc_info:
            LiveKitTokenRequest(thread_id=thread_id)  # type: ignore[call-arg]

        assert "session_id" in str(exc_info.value)

    def test_participant_name_min_length(self) -> None:
        """Test participant_name minimum length validation."""
        thread_id = uuid4()
        session_id = uuid4()

        with pytest.raises(ValidationError) as exc_info:
            LiveKitTokenRequest(
                thread_id=thread_id,
                session_id=session_id,
                participant_name="",  # Empty string
            )

        assert "participant_name" in str(exc_info.value)

    def test_participant_name_max_length(self) -> None:
        """Test participant_name maximum length validation."""
        thread_id = uuid4()
        session_id = uuid4()

        with pytest.raises(ValidationError) as exc_info:
            LiveKitTokenRequest(
                thread_id=thread_id,
                session_id=session_id,
                participant_name="a" * 101,  # Too long
            )

        assert "participant_name" in str(exc_info.value)

    def test_ttl_minimum(self) -> None:
        """Test ttl_seconds minimum value validation."""
        thread_id = uuid4()
        session_id = uuid4()

        with pytest.raises(ValidationError) as exc_info:
            LiveKitTokenRequest(
                thread_id=thread_id,
                session_id=session_id,
                ttl_seconds=10,  # Below minimum of 30
            )

        assert "ttl_seconds" in str(exc_info.value)

    def test_ttl_maximum(self) -> None:
        """Test ttl_seconds maximum value validation."""
        thread_id = uuid4()
        session_id = uuid4()

        with pytest.raises(ValidationError) as exc_info:
            LiveKitTokenRequest(
                thread_id=thread_id,
                session_id=session_id,
                ttl_seconds=7200,  # Above maximum of 3600
            )

        assert "ttl_seconds" in str(exc_info.value)

    def test_ttl_boundary_values(self) -> None:
        """Test ttl_seconds boundary values are valid."""
        thread_id = uuid4()
        session_id = uuid4()

        # Minimum valid
        request_min = LiveKitTokenRequest(
            thread_id=thread_id,
            session_id=session_id,
            ttl_seconds=30,
        )
        assert request_min.ttl_seconds == 30

        # Maximum valid
        request_max = LiveKitTokenRequest(
            thread_id=thread_id,
            session_id=session_id,
            ttl_seconds=3600,
        )
        assert request_max.ttl_seconds == 3600


class TestLiveKitTokenResponse:
    """Tests for LiveKitTokenResponse schema."""

    def test_valid_response(self) -> None:
        """Test valid response with all fields."""
        response = LiveKitTokenResponse(
            token="jwt-token-here",
            room_name="pl-thread-123",
            livekit_url="wss://test.livekit.cloud",
            expires_in=120,
        )

        assert response.token == "jwt-token-here"
        assert response.room_name == "pl-thread-123"
        assert response.livekit_url == "wss://test.livekit.cloud"
        assert response.expires_in == 120

    def test_missing_fields_raise(self) -> None:
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            LiveKitTokenResponse()  # type: ignore[call-arg]


class TestLiveKitConfig:
    """Tests for LiveKitConfig schema."""

    def test_enabled_with_url(self) -> None:
        """Test config with enabled=True and URL."""
        config = LiveKitConfig(
            enabled=True,
            url="wss://test.livekit.cloud",
        )

        assert config.enabled is True
        assert config.url == "wss://test.livekit.cloud"

    def test_disabled_without_url(self) -> None:
        """Test config with enabled=False and no URL."""
        config = LiveKitConfig(
            enabled=False,
            url=None,
        )

        assert config.enabled is False
        assert config.url is None

    def test_url_default_none(self) -> None:
        """Test that URL defaults to None."""
        config = LiveKitConfig(enabled=False)

        assert config.url is None
