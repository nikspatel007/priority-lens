"""Clerk authentication configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClerkConfig(BaseSettings):
    """Clerk authentication configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CLERK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Clerk configuration
    publishable_key: str = Field(
        default="",
        description="Clerk publishable key (pk_test_... or pk_live_...)",
    )
    secret_key: str = Field(
        default="",
        description="Clerk secret key (sk_test_... or sk_live_...)",
    )

    # JWT validation
    issuer: str = Field(
        default="",
        description="Expected JWT issuer (e.g., https://clerk.your-domain.com)",
    )
    jwks_url: str = Field(
        default="",
        description="JWKS endpoint URL for key retrieval",
    )
    audience: str = Field(
        default="",
        description="Expected JWT audience (optional)",
    )

    # Token settings
    token_leeway_seconds: int = Field(
        default=30,
        ge=0,
        le=300,
        description="Leeway in seconds for token expiration",
    )

    # API key support for service-to-service calls
    api_keys: list[str] = Field(
        default_factory=list,
        description="List of valid API keys for service authentication",
    )

    @property
    def is_configured(self) -> bool:
        """Check if Clerk is properly configured."""
        return bool(self.secret_key and self.issuer)

    @property
    def effective_jwks_url(self) -> str:
        """Get JWKS URL, deriving from issuer if not explicitly set."""
        if self.jwks_url:
            return self.jwks_url
        if self.issuer:
            # Standard Clerk JWKS path
            return f"{self.issuer.rstrip('/')}/.well-known/jwks.json"
        return ""


@lru_cache
def get_clerk_config() -> ClerkConfig:
    """Get cached Clerk configuration instance."""
    return ClerkConfig()
