"""Tests for CORS middleware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rl_emails.api.config import APIConfig
from rl_emails.api.middleware.cors import setup_cors


class TestSetupCors:
    """Tests for setup_cors function."""

    def test_adds_cors_middleware(self) -> None:
        """Test that CORS middleware is added to app."""
        mock_app = MagicMock()
        config = APIConfig(cors_origins=["http://localhost:3000", "https://example.com"])

        setup_cors(mock_app, config)

        mock_app.add_middleware.assert_called_once()

    def test_cors_middleware_config(self) -> None:
        """Test CORS middleware configuration."""
        mock_app = MagicMock()
        config = APIConfig(cors_origins=["http://localhost:3000"])

        with patch("rl_emails.api.middleware.cors.CORSMiddleware"):
            setup_cors(mock_app, config)

            call_kwargs = mock_app.add_middleware.call_args[1]
            assert call_kwargs["allow_origins"] == ["http://localhost:3000"]
            assert call_kwargs["allow_credentials"] is True
            assert "GET" in call_kwargs["allow_methods"]
            assert "POST" in call_kwargs["allow_methods"]
            assert call_kwargs["allow_headers"] == ["*"]

    def test_exposes_correlation_headers(self) -> None:
        """Test that correlation ID headers are exposed."""
        mock_app = MagicMock()
        config = APIConfig()

        setup_cors(mock_app, config)

        call_kwargs = mock_app.add_middleware.call_args[1]
        assert "X-Request-ID" in call_kwargs["expose_headers"]
        assert "X-Correlation-ID" in call_kwargs["expose_headers"]
