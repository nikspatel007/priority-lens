"""Tests for FastAPI application factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from rl_emails.api.config import APIConfig
from rl_emails.api.main import create_app, lifespan, run_server


class TestCreateApp:
    """Tests for create_app function."""

    def test_returns_fastapi_app(self) -> None:
        """Test that create_app returns a FastAPI application."""
        config = APIConfig(environment="development")

        with (
            patch("rl_emails.api.main.setup_cors"),
            patch("rl_emails.api.main.setup_error_handlers"),
            patch("rl_emails.api.main.setup_logging"),
            patch("rl_emails.api.main.setup_rate_limit"),
        ):
            app = create_app(config)

        assert isinstance(app, FastAPI)

    def test_app_metadata(self) -> None:
        """Test that app has correct metadata."""
        config = APIConfig(environment="development")

        with (
            patch("rl_emails.api.main.setup_cors"),
            patch("rl_emails.api.main.setup_error_handlers"),
            patch("rl_emails.api.main.setup_logging"),
            patch("rl_emails.api.main.setup_rate_limit"),
        ):
            app = create_app(config)

        assert app.title == "rl-emails API"
        assert app.version == "2.0.0"

    def test_docs_enabled_in_development(self) -> None:
        """Test that docs are enabled in development."""
        config = APIConfig(environment="development")

        with (
            patch("rl_emails.api.main.setup_cors"),
            patch("rl_emails.api.main.setup_error_handlers"),
            patch("rl_emails.api.main.setup_logging"),
            patch("rl_emails.api.main.setup_rate_limit"),
        ):
            app = create_app(config)

        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"

    def test_docs_disabled_in_production(self) -> None:
        """Test that docs are disabled in production."""
        config = APIConfig(environment="production")

        with (
            patch("rl_emails.api.main.setup_cors"),
            patch("rl_emails.api.main.setup_error_handlers"),
            patch("rl_emails.api.main.setup_logging"),
            patch("rl_emails.api.main.setup_rate_limit"),
        ):
            app = create_app(config)

        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.openapi_url is None

    def test_config_stored_in_state(self) -> None:
        """Test that config is stored in app state."""
        config = APIConfig()

        with (
            patch("rl_emails.api.main.setup_cors"),
            patch("rl_emails.api.main.setup_error_handlers"),
            patch("rl_emails.api.main.setup_logging"),
            patch("rl_emails.api.main.setup_rate_limit"),
        ):
            app = create_app(config)

        assert app.state.config is config

    def test_uses_default_config_when_none(self) -> None:
        """Test that default config is used when none provided."""
        with (
            patch("rl_emails.api.main.get_api_config") as mock_get_config,
            patch("rl_emails.api.main.setup_cors"),
            patch("rl_emails.api.main.setup_error_handlers"),
            patch("rl_emails.api.main.setup_logging"),
            patch("rl_emails.api.main.setup_rate_limit"),
        ):
            mock_config = APIConfig()
            mock_get_config.return_value = mock_config

            app = create_app(None)

            mock_get_config.assert_called_once()
            assert app.state.config is mock_config

    def test_middleware_setup_called(self) -> None:
        """Test that all middleware setup functions are called."""
        config = APIConfig()

        with (
            patch("rl_emails.api.main.setup_cors") as mock_cors,
            patch("rl_emails.api.main.setup_error_handlers") as mock_errors,
            patch("rl_emails.api.main.setup_logging") as mock_logging,
            patch("rl_emails.api.main.setup_rate_limit") as mock_rate_limit,
        ):
            create_app(config)

            mock_cors.assert_called_once()
            mock_errors.assert_called_once()
            mock_logging.assert_called_once()
            mock_rate_limit.assert_called_once()

    def test_health_router_included(self) -> None:
        """Test that health router is included."""
        config = APIConfig()

        with (
            patch("rl_emails.api.main.setup_cors"),
            patch("rl_emails.api.main.setup_error_handlers"),
            patch("rl_emails.api.main.setup_logging"),
            patch("rl_emails.api.main.setup_rate_limit"),
        ):
            app = create_app(config)

        # Check that health routes are registered
        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths or any("/health" in p for p in route_paths)


class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    async def test_startup_connects_database(self) -> None:
        """Test that database is connected on startup."""
        config = APIConfig()
        mock_app = MagicMock(spec=FastAPI)
        mock_app.state = MagicMock()
        mock_app.state.config = config

        with (
            patch("rl_emails.api.main.Database") as mock_db_class,
            patch("rl_emails.api.main.set_database") as mock_set_db,
            patch("rl_emails.api.main.logger") as mock_logger,
        ):
            mock_db = AsyncMock()
            mock_db_class.return_value = mock_db
            mock_logger.ainfo = AsyncMock()

            async with lifespan(mock_app):
                mock_db.connect.assert_called_once()
                mock_set_db.assert_called_once_with(mock_db)

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_database(self) -> None:
        """Test that database is disconnected on shutdown."""
        config = APIConfig()
        mock_app = MagicMock(spec=FastAPI)
        mock_app.state = MagicMock()
        mock_app.state.config = config

        with (
            patch("rl_emails.api.main.Database") as mock_db_class,
            patch("rl_emails.api.main.set_database") as mock_set_db,
            patch("rl_emails.api.main.logger") as mock_logger,
        ):
            mock_db = AsyncMock()
            mock_db_class.return_value = mock_db
            mock_logger.ainfo = AsyncMock()

            async with lifespan(mock_app):
                pass

            # After context exits, disconnect should be called
            mock_db.disconnect.assert_called_once()
            # set_database(None) should be called on shutdown
            assert mock_set_db.call_count == 2
            mock_set_db.assert_called_with(None)

    @pytest.mark.asyncio
    async def test_logs_startup_and_shutdown(self) -> None:
        """Test that startup and shutdown are logged."""
        config = APIConfig()
        mock_app = MagicMock(spec=FastAPI)
        mock_app.state = MagicMock()
        mock_app.state.config = config

        with (
            patch("rl_emails.api.main.Database") as mock_db_class,
            patch("rl_emails.api.main.set_database"),
            patch("rl_emails.api.main.logger") as mock_logger,
        ):
            mock_db = AsyncMock()
            mock_db_class.return_value = mock_db
            mock_logger.ainfo = AsyncMock()

            async with lifespan(mock_app):
                pass

            # Should log startup and shutdown events
            assert mock_logger.ainfo.call_count >= 4

    @pytest.mark.asyncio
    async def test_shutdown_without_db(self) -> None:
        """Test shutdown when db attribute is missing."""
        config = APIConfig()
        mock_app = MagicMock(spec=FastAPI)
        mock_app.state = MagicMock()
        mock_app.state.config = config

        # Simulate db not being set (no db attribute)
        del mock_app.state.db

        with (
            patch("rl_emails.api.main.Database") as mock_db_class,
            patch("rl_emails.api.main.set_database"),
            patch("rl_emails.api.main.logger") as mock_logger,
        ):
            mock_db = AsyncMock()
            mock_db_class.return_value = mock_db
            mock_logger.ainfo = AsyncMock()

            # Remove the db after startup to test shutdown branch
            async with lifespan(mock_app):
                # Remove db to simulate it not being present
                mock_app.state.db = None

            # Should complete without error even though db is None


class TestRunServer:
    """Tests for run_server function."""

    def test_runs_uvicorn(self) -> None:
        """Test that uvicorn is started with correct config."""
        config = APIConfig(host="127.0.0.1", port=9000, environment="development")

        with patch("uvicorn.run") as mock_uvicorn_run:
            run_server(config)

            mock_uvicorn_run.assert_called_once()
            call_kwargs = mock_uvicorn_run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 9000
            assert call_kwargs["reload"] is True

    def test_uses_default_config_when_none(self) -> None:
        """Test that default config is used when none provided."""
        with (
            patch("rl_emails.api.main.get_api_config") as mock_get_config,
            patch("uvicorn.run") as mock_uvicorn_run,
        ):
            mock_config = APIConfig()
            mock_get_config.return_value = mock_config

            run_server(None)

            mock_get_config.assert_called_once()
            mock_uvicorn_run.assert_called_once()

    def test_reload_disabled_in_production(self) -> None:
        """Test that reload is disabled in production."""
        config = APIConfig(environment="production")

        with patch("uvicorn.run") as mock_uvicorn_run:
            run_server(config)

            call_kwargs = mock_uvicorn_run.call_args[1]
            assert call_kwargs["reload"] is False
