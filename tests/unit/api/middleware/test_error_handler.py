"""Tests for error handler middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from rl_emails.api.exceptions import APIError, NotFoundError, ValidationError
from rl_emails.api.middleware.error_handler import (
    api_error_handler,
    http_exception_handler,
    setup_error_handlers,
    unhandled_exception_handler,
    validation_error_handler,
)


class TestAPIErrorHandler:
    """Tests for api_error_handler function."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/api/test"
        return request

    @pytest.mark.asyncio
    async def test_handles_api_error(self, mock_request: MagicMock) -> None:
        """Test handling of APIError."""
        error = NotFoundError(resource="User", resource_id="123")

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            response = await api_error_handler(mock_request, error)

        assert response.status_code == 404
        assert response.media_type == "application/problem+json"

    @pytest.mark.asyncio
    async def test_response_contains_problem_detail(self, mock_request: MagicMock) -> None:
        """Test that response contains Problem Detail format."""
        error = APIError(
            title="Test Error",
            detail="Test detail",
            status=418,
            error_type="/errors/test",
        )

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            response = await api_error_handler(mock_request, error)

        # Parse response body
        import json

        body = json.loads(response.body.decode())

        assert body["type"] == "/errors/test"
        assert body["title"] == "Test Error"
        assert body["status"] == 418
        assert body["detail"] == "Test detail"

    @pytest.mark.asyncio
    async def test_logs_error(self, mock_request: MagicMock) -> None:
        """Test that errors are logged."""
        error = ValidationError(detail="Validation failed")

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            await api_error_handler(mock_request, error)

            mock_logger.awarning.assert_called_once()
            call_kwargs = mock_logger.awarning.call_args[1]
            assert call_kwargs["error_type"] == "/errors/validation"
            assert call_kwargs["status"] == 422


class TestHTTPExceptionHandler:
    """Tests for http_exception_handler function."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/api/test"
        return request

    @pytest.mark.asyncio
    async def test_handles_http_exception(self, mock_request: MagicMock) -> None:
        """Test handling of Starlette HTTPException."""
        error = StarletteHTTPException(status_code=404, detail="Not found")

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            response = await http_exception_handler(mock_request, error)

        assert response.status_code == 404
        assert response.media_type == "application/problem+json"

    @pytest.mark.asyncio
    async def test_response_format(self, mock_request: MagicMock) -> None:
        """Test Problem Detail response format."""
        error = StarletteHTTPException(status_code=500, detail="Internal error")

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            response = await http_exception_handler(mock_request, error)

        import json

        body = json.loads(response.body.decode())

        assert body["type"] == "/errors/http"
        assert body["status"] == 500
        assert body["detail"] == "Internal error"


class TestValidationErrorHandler:
    """Tests for validation_error_handler function."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/api/test"
        return request

    @pytest.mark.asyncio
    async def test_handles_validation_error(self, mock_request: MagicMock) -> None:
        """Test handling of RequestValidationError."""
        # Create mock validation error
        errors = [
            {"loc": ("body", "email"), "msg": "Invalid email", "type": "value_error"},
            {"loc": ("body", "age"), "msg": "Must be positive", "type": "value_error"},
        ]
        exc = RequestValidationError(errors=errors)

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            response = await validation_error_handler(mock_request, exc)

        assert response.status_code == 422
        assert response.media_type == "application/problem+json"

    @pytest.mark.asyncio
    async def test_formats_validation_errors(self, mock_request: MagicMock) -> None:
        """Test that validation errors are properly formatted."""
        errors = [
            {"loc": ("body", "email"), "msg": "Invalid email", "type": "value_error"},
        ]
        exc = RequestValidationError(errors=errors)

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            response = await validation_error_handler(mock_request, exc)

        import json

        body = json.loads(response.body.decode())

        assert body["type"] == "/errors/validation"
        assert body["title"] == "Validation Error"
        assert body["status"] == 422
        assert len(body["errors"]) == 1
        assert body["errors"][0]["field"] == "body.email"
        assert body["errors"][0]["message"] == "Invalid email"


class TestUnhandledExceptionHandler:
    """Tests for unhandled_exception_handler function."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/api/test"
        return request

    @pytest.mark.asyncio
    async def test_handles_unhandled_exception(self, mock_request: MagicMock) -> None:
        """Test handling of unexpected exceptions."""
        error = Exception("Unexpected error")

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.aexception = AsyncMock()
            response = await unhandled_exception_handler(mock_request, error)

        assert response.status_code == 500
        assert response.media_type == "application/problem+json"

    @pytest.mark.asyncio
    async def test_generic_error_response(self, mock_request: MagicMock) -> None:
        """Test that generic error doesn't leak details."""
        error = ValueError("Sensitive information")

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.aexception = AsyncMock()
            response = await unhandled_exception_handler(mock_request, error)

        import json

        body = json.loads(response.body.decode())

        assert body["type"] == "/errors/internal"
        assert body["title"] == "Internal Server Error"
        assert body["detail"] == "An unexpected error occurred"
        # Should not contain "Sensitive information"

    @pytest.mark.asyncio
    async def test_logs_exception(self, mock_request: MagicMock) -> None:
        """Test that exception is logged."""
        error = RuntimeError("Test error")

        with patch("rl_emails.api.middleware.error_handler.logger") as mock_logger:
            mock_logger.aexception = AsyncMock()
            await unhandled_exception_handler(mock_request, error)

            mock_logger.aexception.assert_called_once()
            call_kwargs = mock_logger.aexception.call_args[1]
            assert call_kwargs["exc_type"] == "RuntimeError"


class TestSetupErrorHandlers:
    """Tests for setup_error_handlers function."""

    def test_registers_handlers(self) -> None:
        """Test that all handlers are registered."""
        app = FastAPI()

        setup_error_handlers(app)

        # Check that handlers were added
        assert len(app.exception_handlers) >= 4
