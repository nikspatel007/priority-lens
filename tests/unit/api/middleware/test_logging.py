"""Tests for logging middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from priority_lens.api.config import APIConfig
from priority_lens.api.middleware.logging import (
    RequestLoggingMiddleware,
    add_correlation_id,
    configure_structlog,
    request_id_var,
    setup_logging,
    user_id_var,
)


class TestAddCorrelationId:
    """Tests for add_correlation_id processor."""

    def test_adds_correlation_id(self) -> None:
        """Test that correlation ID is added from context."""
        event_dict: dict[str, object] = {"event": "test"}

        with patch("priority_lens.api.middleware.logging.correlation_id") as mock_cid:
            mock_cid.get.return_value = "test-correlation-id"
            result = add_correlation_id(MagicMock(), "info", event_dict)

        assert result["correlation_id"] == "test-correlation-id"

    def test_adds_request_id(self) -> None:
        """Test that request ID is added from context var."""
        event_dict: dict[str, object] = {"event": "test"}

        with patch("priority_lens.api.middleware.logging.correlation_id") as mock_cid:
            mock_cid.get.return_value = None
            request_id_var.set("test-request-id")
            result = add_correlation_id(MagicMock(), "info", event_dict)
            request_id_var.set(None)

        assert result["request_id"] == "test-request-id"

    def test_adds_user_id(self) -> None:
        """Test that user ID is added from context var."""
        event_dict: dict[str, object] = {"event": "test"}

        with patch("priority_lens.api.middleware.logging.correlation_id") as mock_cid:
            mock_cid.get.return_value = None
            user_id_var.set("test-user-id")
            result = add_correlation_id(MagicMock(), "info", event_dict)
            user_id_var.set(None)

        assert result["user_id"] == "test-user-id"

    def test_no_ids_when_not_set(self) -> None:
        """Test that IDs are not added when not set."""
        event_dict: dict[str, object] = {"event": "test"}

        with patch("priority_lens.api.middleware.logging.correlation_id") as mock_cid:
            mock_cid.get.return_value = None
            request_id_var.set(None)
            user_id_var.set(None)
            result = add_correlation_id(MagicMock(), "info", event_dict)

        assert "correlation_id" not in result
        assert "request_id" not in result
        assert "user_id" not in result


class TestConfigureStructlog:
    """Tests for configure_structlog function."""

    def test_json_format(self) -> None:
        """Test configuring with JSON format."""
        with patch("priority_lens.api.middleware.logging.structlog") as mock_structlog:
            configure_structlog(json_format=True, log_level="INFO")

            mock_structlog.configure.assert_called_once()

    def test_console_format(self) -> None:
        """Test configuring with console format."""
        with patch("priority_lens.api.middleware.logging.structlog") as mock_structlog:
            configure_structlog(json_format=False, log_level="DEBUG")

            mock_structlog.configure.assert_called_once()

    def test_log_levels(self) -> None:
        """Test different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            with patch("priority_lens.api.middleware.logging.structlog") as mock_structlog:
                configure_structlog(json_format=True, log_level=level)
                mock_structlog.configure.assert_called_once()


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware class."""

    @pytest.fixture
    def middleware(self) -> RequestLoggingMiddleware:
        """Create middleware instance."""
        app = MagicMock()
        return RequestLoggingMiddleware(app)

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock request."""
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.url.query = ""
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_dispatch_success(
        self, middleware: RequestLoggingMiddleware, mock_request: MagicMock
    ) -> None:
        """Test successful request dispatch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def call_next(request: MagicMock) -> MagicMock:
            return mock_response

        with patch("priority_lens.api.middleware.logging.logger") as mock_logger:
            mock_logger.ainfo = AsyncMock()
            response = await middleware.dispatch(mock_request, call_next)

        assert response is mock_response
        assert mock_logger.ainfo.call_count == 2  # request_started and request_completed

    @pytest.mark.asyncio
    async def test_dispatch_adds_request_id_header(
        self, middleware: RequestLoggingMiddleware, mock_request: MagicMock
    ) -> None:
        """Test that X-Request-ID header is added to response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def call_next(request: MagicMock) -> MagicMock:
            return mock_response

        with patch("priority_lens.api.middleware.logging.logger") as mock_logger:
            mock_logger.ainfo = AsyncMock()
            response = await middleware.dispatch(mock_request, call_next)

        assert "X-Request-ID" in response.headers

    @pytest.mark.asyncio
    async def test_dispatch_extracts_user_id(
        self, middleware: RequestLoggingMiddleware, mock_request: MagicMock
    ) -> None:
        """Test that user ID is extracted from headers."""
        mock_request.headers = {"X-User-ID": "test-user"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        captured_user_id = None

        async def call_next(request: MagicMock) -> MagicMock:
            nonlocal captured_user_id
            captured_user_id = user_id_var.get()
            return mock_response

        with patch("priority_lens.api.middleware.logging.logger") as mock_logger:
            mock_logger.ainfo = AsyncMock()
            await middleware.dispatch(mock_request, call_next)

        assert captured_user_id == "test-user"

    @pytest.mark.asyncio
    async def test_dispatch_logs_on_error(
        self, middleware: RequestLoggingMiddleware, mock_request: MagicMock
    ) -> None:
        """Test that errors are logged."""

        async def call_next(request: MagicMock) -> MagicMock:
            raise ValueError("Test error")

        with (
            patch("priority_lens.api.middleware.logging.logger") as mock_logger,
            pytest.raises(ValueError),
        ):
            mock_logger.ainfo = AsyncMock()
            mock_logger.aerror = AsyncMock()
            await middleware.dispatch(mock_request, call_next)

        mock_logger.aerror.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_cleans_up_context(
        self, middleware: RequestLoggingMiddleware, mock_request: MagicMock
    ) -> None:
        """Test that context vars are cleaned up after request."""
        mock_request.headers = {"X-User-ID": "test-user"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def call_next(request: MagicMock) -> MagicMock:
            return mock_response

        with patch("priority_lens.api.middleware.logging.logger") as mock_logger:
            mock_logger.ainfo = AsyncMock()
            await middleware.dispatch(mock_request, call_next)

        # After dispatch, context should be cleaned
        assert request_id_var.get() is None
        assert user_id_var.get() is None

    @pytest.mark.asyncio
    async def test_dispatch_logs_query_string(
        self, middleware: RequestLoggingMiddleware, mock_request: MagicMock
    ) -> None:
        """Test that query string is logged when present."""
        mock_request.url.query = "page=1&limit=10"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def call_next(request: MagicMock) -> MagicMock:
            return mock_response

        with patch("priority_lens.api.middleware.logging.logger") as mock_logger:
            mock_logger.ainfo = AsyncMock()
            await middleware.dispatch(mock_request, call_next)

            # Check first call has query param
            first_call = mock_logger.ainfo.call_args_list[0]
            assert first_call[1]["query"] == "page=1&limit=10"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging(self) -> None:
        """Test that logging is set up correctly."""
        mock_app = MagicMock()
        config = APIConfig(log_json=True, log_level="INFO")

        with patch("priority_lens.api.middleware.logging.configure_structlog") as mock_configure:
            setup_logging(mock_app, config)

            mock_configure.assert_called_once_with(
                json_format=True,
                log_level="INFO",
            )

    def test_adds_middlewares(self) -> None:
        """Test that middlewares are added."""
        mock_app = MagicMock()
        config = APIConfig()

        with patch("priority_lens.api.middleware.logging.configure_structlog"):
            setup_logging(mock_app, config)

        # Should add CorrelationIdMiddleware and RequestLoggingMiddleware
        assert mock_app.add_middleware.call_count == 2
