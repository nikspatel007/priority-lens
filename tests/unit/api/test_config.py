"""Tests for API configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from rl_emails.api.config import APIConfig, get_api_config


class TestAPIConfig:
    """Tests for APIConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = APIConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.environment == "development"
        assert config.db_pool_size == 10
        assert config.db_max_overflow == 5
        assert config.db_pool_recycle == 1800
        assert config.rate_limit_requests == 100
        assert config.rate_limit_window == 60
        assert config.log_level == "INFO"
        assert config.log_json is True

    def test_environment_values(self) -> None:
        """Test loading from environment variables."""
        env_vars = {
            "API_HOST": "127.0.0.1",
            "API_PORT": "9000",
            "API_DEBUG": "true",
            "API_ENVIRONMENT": "production",
            "API_DATABASE_URL": "postgresql://prod:prod@db:5432/prod",
            "API_DB_POOL_SIZE": "20",
            "API_DB_MAX_OVERFLOW": "10",
            "API_DB_POOL_RECYCLE": "3600",
            "API_RATE_LIMIT_REQUESTS": "200",
            "API_RATE_LIMIT_WINDOW": "120",
            "API_LOG_LEVEL": "DEBUG",
            "API_LOG_JSON": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = APIConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.debug is True
        assert config.environment == "production"
        assert config.database_url == "postgresql+asyncpg://prod:prod@db:5432/prod"
        assert config.db_pool_size == 20
        assert config.db_max_overflow == 10
        assert config.db_pool_recycle == 3600
        assert config.rate_limit_requests == 200
        assert config.rate_limit_window == 120
        assert config.log_level == "DEBUG"
        assert config.log_json is False

    def test_database_url_conversion(self) -> None:
        """Test that postgresql:// is converted to postgresql+asyncpg://."""
        with patch.dict(
            os.environ,
            {"API_DATABASE_URL": "postgresql://user:pass@host:5432/db"},
            clear=False,
        ):
            config = APIConfig()

        assert config.database_url == "postgresql+asyncpg://user:pass@host:5432/db"

    def test_database_url_already_async(self) -> None:
        """Test that postgresql+asyncpg:// URLs are not modified."""
        with patch.dict(
            os.environ,
            {"API_DATABASE_URL": "postgresql+asyncpg://user:pass@host:5432/db"},
            clear=False,
        ):
            config = APIConfig()

        assert config.database_url == "postgresql+asyncpg://user:pass@host:5432/db"

    def test_is_production(self) -> None:
        """Test is_production property."""
        with patch.dict(os.environ, {"API_ENVIRONMENT": "production"}, clear=False):
            config = APIConfig()
            assert config.is_production is True
            assert config.is_development is False

    def test_is_development(self) -> None:
        """Test is_development property."""
        with patch.dict(os.environ, {"API_ENVIRONMENT": "development"}, clear=False):
            config = APIConfig()
            assert config.is_development is True
            assert config.is_production is False

    def test_is_staging(self) -> None:
        """Test staging environment."""
        with patch.dict(os.environ, {"API_ENVIRONMENT": "staging"}, clear=False):
            config = APIConfig()
            assert config.is_development is False
            assert config.is_production is False

    def test_port_validation_min(self) -> None:
        """Test that port must be >= 1."""
        with patch.dict(os.environ, {"API_PORT": "0"}, clear=False):
            with pytest.raises(ValueError):
                APIConfig()

    def test_port_validation_max(self) -> None:
        """Test that port must be <= 65535."""
        with patch.dict(os.environ, {"API_PORT": "65536"}, clear=False):
            with pytest.raises(ValueError):
                APIConfig()

    def test_db_pool_size_validation(self) -> None:
        """Test db_pool_size validation."""
        with patch.dict(os.environ, {"API_DB_POOL_SIZE": "0"}, clear=False):
            with pytest.raises(ValueError):
                APIConfig()

    def test_cors_origins_default(self) -> None:
        """Test default CORS origins."""
        config = APIConfig()
        assert config.cors_origins == ["http://localhost:3000"]

    def test_cors_origins_from_env(self) -> None:
        """Test CORS origins from environment."""
        with patch.dict(
            os.environ,
            {"API_CORS_ORIGINS": '["https://example.com", "https://app.example.com"]'},
            clear=False,
        ):
            config = APIConfig()
            assert "https://example.com" in config.cors_origins


class TestGetAPIConfig:
    """Tests for get_api_config function."""

    def test_returns_config(self) -> None:
        """Test that get_api_config returns an APIConfig instance."""
        # Clear the cache to test fresh instance
        get_api_config.cache_clear()

        config = get_api_config()
        assert isinstance(config, APIConfig)

    def test_cached(self) -> None:
        """Test that config is cached."""
        get_api_config.cache_clear()

        config1 = get_api_config()
        config2 = get_api_config()

        assert config1 is config2
