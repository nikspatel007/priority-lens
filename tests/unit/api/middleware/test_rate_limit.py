"""Tests for rate limit middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.api.config import APIConfig
from rl_emails.api.middleware.rate_limit import (
    _get_request_identifier,
    get_limiter,
    rate_limit_exceeded_handler,
    setup_rate_limit,
)


class TestGetRequestIdentifier:
    """Tests for _get_request_identifier function."""

    def test_returns_user_id_when_present(self) -> None:
        """Test that user ID is used when present in headers."""
        mock_request = MagicMock()
        mock_request.headers = {"X-User-ID": "user-123"}

        result = _get_request_identifier(mock_request)

        assert result == "user:user-123"

    def test_returns_ip_when_no_user(self) -> None:
        """Test that IP is used when no user ID header."""
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.1"

        with patch("rl_emails.api.middleware.rate_limit.get_remote_address") as mock_get_ip:
            mock_get_ip.return_value = "192.168.1.1"
            result = _get_request_identifier(mock_request)

        assert result == "192.168.1.1"


class TestRateLimitExceededHandler:
    """Tests for rate_limit_exceeded_handler function."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        return request

    @pytest.fixture
    def mock_exc(self) -> MagicMock:
        """Create mock RateLimitExceeded exception."""
        exc = MagicMock()
        exc.detail = "Rate limit exceeded"
        return exc

    @pytest.mark.asyncio
    async def test_returns_429_response(self, mock_request: MagicMock, mock_exc: MagicMock) -> None:
        """Test that handler returns 429 status."""
        with (
            patch("rl_emails.api.middleware.rate_limit.logger") as mock_logger,
            patch(
                "rl_emails.api.middleware.rate_limit._get_request_identifier",
                return_value="test-id",
            ),
        ):
            mock_logger.awarning = AsyncMock()
            response = await rate_limit_exceeded_handler(mock_request, mock_exc)

        assert response.status_code == 429
        assert response.media_type == "application/problem+json"

    @pytest.mark.asyncio
    async def test_response_contains_problem_detail(
        self, mock_request: MagicMock, mock_exc: MagicMock
    ) -> None:
        """Test that response contains Problem Detail format."""
        with (
            patch("rl_emails.api.middleware.rate_limit.logger") as mock_logger,
            patch(
                "rl_emails.api.middleware.rate_limit._get_request_identifier",
                return_value="test-id",
            ),
        ):
            mock_logger.awarning = AsyncMock()
            response = await rate_limit_exceeded_handler(mock_request, mock_exc)

        import json

        body = json.loads(response.body.decode())

        assert body["type"] == "/errors/rate-limit"
        assert body["title"] == "Too Many Requests"
        assert body["status"] == 429

    @pytest.mark.asyncio
    async def test_extracts_retry_after(self, mock_request: MagicMock) -> None:
        """Test that retry_after is extracted from exception detail."""
        mock_exc = MagicMock()
        mock_exc.detail = "Rate limit exceeded: 60 per minute"

        with (
            patch("rl_emails.api.middleware.rate_limit.logger") as mock_logger,
            patch(
                "rl_emails.api.middleware.rate_limit._get_request_identifier",
                return_value="test-id",
            ),
        ):
            mock_logger.awarning = AsyncMock()
            response = await rate_limit_exceeded_handler(mock_request, mock_exc)

        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_logs_rate_limit_exceeded(
        self, mock_request: MagicMock, mock_exc: MagicMock
    ) -> None:
        """Test that rate limit exceeded is logged."""
        with (
            patch("rl_emails.api.middleware.rate_limit.logger") as mock_logger,
            patch(
                "rl_emails.api.middleware.rate_limit._get_request_identifier",
                return_value="user:test-123",
            ),
        ):
            mock_logger.awarning = AsyncMock()
            await rate_limit_exceeded_handler(mock_request, mock_exc)

            mock_logger.awarning.assert_called_once()
            call_kwargs = mock_logger.awarning.call_args[1]
            assert call_kwargs["identifier"] == "user:test-123"

    @pytest.mark.asyncio
    async def test_no_retry_after_without_number(self, mock_request: MagicMock) -> None:
        """Test that no Retry-After header when no number in detail."""
        mock_exc = MagicMock()
        mock_exc.detail = "Rate limit exceeded"  # No number

        with (
            patch("rl_emails.api.middleware.rate_limit.logger") as mock_logger,
            patch(
                "rl_emails.api.middleware.rate_limit._get_request_identifier",
                return_value="test-id",
            ),
        ):
            mock_logger.awarning = AsyncMock()
            response = await rate_limit_exceeded_handler(mock_request, mock_exc)

        # Should not have Retry-After since no number was found
        assert "Retry-After" not in response.headers

    @pytest.mark.asyncio
    async def test_handles_detail_not_string(self, mock_request: MagicMock) -> None:
        """Test that non-string detail is handled gracefully."""
        mock_exc = MagicMock()
        mock_exc.detail = 12345  # Not a string

        with (
            patch("rl_emails.api.middleware.rate_limit.logger") as mock_logger,
            patch(
                "rl_emails.api.middleware.rate_limit._get_request_identifier",
                return_value="test-id",
            ),
        ):
            mock_logger.awarning = AsyncMock()
            response = await rate_limit_exceeded_handler(mock_request, mock_exc)

        # Should complete without error
        assert response.status_code == 429


class TestSetupRateLimit:
    """Tests for setup_rate_limit function."""

    def test_creates_limiter(self) -> None:
        """Test that limiter is created."""
        mock_app = MagicMock()
        config = APIConfig(rate_limit_requests=100, rate_limit_window=60)

        setup_rate_limit(mock_app, config)

        assert mock_app.state.limiter is not None

    def test_adds_exception_handler(self) -> None:
        """Test that exception handler is added."""
        mock_app = MagicMock()
        config = APIConfig()

        setup_rate_limit(mock_app, config)

        mock_app.add_exception_handler.assert_called_once()


class TestGetLimiter:
    """Tests for get_limiter function."""

    def test_raises_when_not_configured(self) -> None:
        """Test that get_limiter raises when not configured."""
        # Reset the global limiter
        import rl_emails.api.middleware.rate_limit as rate_limit_module

        original = rate_limit_module.limiter
        rate_limit_module.limiter = None

        try:
            with pytest.raises(RuntimeError, match="Rate limiter not configured"):
                get_limiter()
        finally:
            rate_limit_module.limiter = original

    def test_returns_limiter_when_configured(self) -> None:
        """Test that limiter is returned when configured."""
        mock_app = MagicMock()
        config = APIConfig()
        setup_rate_limit(mock_app, config)

        limiter = get_limiter()
        assert limiter is not None
