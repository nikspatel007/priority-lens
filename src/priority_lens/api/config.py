"""API configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """API configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5433/gmail_test_30d",
        description="Async database URL",
    )
    db_pool_size: int = Field(default=10, ge=1, le=100, description="Database pool size")
    db_max_overflow: int = Field(default=5, ge=0, le=50, description="Max overflow connections")
    db_pool_recycle: int = Field(default=1800, ge=60, description="Pool recycle time in seconds")

    # Rate limiting
    rate_limit_requests: int = Field(
        default=100, ge=1, description="Requests per rate limit window"
    )
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")

    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    log_json: bool = Field(default=True, description="Use JSON log format")

    # LiveKit configuration
    livekit_api_key: str | None = Field(default=None, description="LiveKit API key")
    livekit_api_secret: str | None = Field(default=None, description="LiveKit API secret")
    livekit_url: str | None = Field(default=None, description="LiveKit server URL (wss://...)")

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL uses asyncpg driver."""
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def has_livekit(self) -> bool:
        """Check if LiveKit credentials are configured."""
        return bool(self.livekit_api_key and self.livekit_api_secret)


@lru_cache
def get_api_config() -> APIConfig:
    """Get cached API configuration instance."""
    return APIConfig()
