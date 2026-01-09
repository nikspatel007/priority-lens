"""Tests for Clerk authentication configuration."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from rl_emails.api.auth.config import ClerkConfig, get_clerk_config


class TestClerkConfig:
    """Tests for ClerkConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClerkConfig(
            publishable_key="",
            secret_key="",
            issuer="",
            jwks_url="",
            audience="",
            api_keys_raw="",  # Explicitly empty
        )

        assert config.publishable_key == ""
        assert config.secret_key == ""
        assert config.issuer == ""
        assert config.jwks_url == ""
        assert config.audience == ""
        assert config.token_leeway_seconds == 30
        assert config.api_keys == []

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ClerkConfig(
            publishable_key="pk_test_xxx",
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            audience="my-app",
            token_leeway_seconds=60,
            api_keys_raw="key1,key2",  # Comma-separated string
        )

        assert config.publishable_key == "pk_test_xxx"
        assert config.secret_key == "sk_test_xxx"
        assert config.issuer == "https://clerk.example.com"
        assert config.jwks_url == "https://clerk.example.com/.well-known/jwks.json"
        assert config.audience == "my-app"
        assert config.token_leeway_seconds == 60
        assert config.api_keys == ["key1", "key2"]

    def test_is_configured_with_secret_and_issuer(self) -> None:
        """Test is_configured returns True when both secret and issuer set."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )

        assert config.is_configured is True

    def test_is_configured_missing_secret(self) -> None:
        """Test is_configured returns False without secret."""
        config = ClerkConfig(
            secret_key="",  # Explicitly empty
            issuer="https://clerk.example.com",
        )

        assert config.is_configured is False

    def test_is_configured_missing_issuer(self) -> None:
        """Test is_configured returns False without issuer."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="",  # Explicitly empty
        )

        assert config.is_configured is False

    def test_is_configured_both_empty(self) -> None:
        """Test is_configured returns False when both empty."""
        config = ClerkConfig(secret_key="", issuer="")  # Explicitly empty

        assert config.is_configured is False

    def test_effective_jwks_url_explicit(self) -> None:
        """Test effective_jwks_url uses explicit URL when provided."""
        config = ClerkConfig(
            issuer="https://clerk.example.com",
            jwks_url="https://custom.jwks.example.com/keys",
        )

        assert config.effective_jwks_url == "https://custom.jwks.example.com/keys"

    def test_effective_jwks_url_derived_from_issuer(self) -> None:
        """Test effective_jwks_url derived from issuer."""
        config = ClerkConfig(
            issuer="https://clerk.example.com",
            jwks_url="",  # Explicitly empty to test derivation
        )

        assert config.effective_jwks_url == "https://clerk.example.com/.well-known/jwks.json"

    def test_effective_jwks_url_strips_trailing_slash(self) -> None:
        """Test effective_jwks_url strips trailing slash from issuer."""
        config = ClerkConfig(
            issuer="https://clerk.example.com/",
            jwks_url="",  # Explicitly empty to test derivation
        )

        assert config.effective_jwks_url == "https://clerk.example.com/.well-known/jwks.json"

    def test_effective_jwks_url_empty_when_no_issuer(self) -> None:
        """Test effective_jwks_url returns empty when no issuer."""
        config = ClerkConfig(issuer="", jwks_url="")  # Explicitly empty

        assert config.effective_jwks_url == ""

    def test_token_leeway_bounds(self) -> None:
        """Test token_leeway_seconds validation."""
        # Valid: within bounds
        config = ClerkConfig(token_leeway_seconds=0)
        assert config.token_leeway_seconds == 0

        config = ClerkConfig(token_leeway_seconds=300)
        assert config.token_leeway_seconds == 300

    def test_token_leeway_below_minimum(self) -> None:
        """Test token_leeway_seconds rejects negative values."""
        with pytest.raises(ValueError):
            ClerkConfig(token_leeway_seconds=-1)

    def test_token_leeway_above_maximum(self) -> None:
        """Test token_leeway_seconds rejects values above 300."""
        with pytest.raises(ValueError):
            ClerkConfig(token_leeway_seconds=301)


class TestGetClerkConfig:
    """Tests for get_clerk_config function."""

    def test_returns_config_instance(self) -> None:
        """Test get_clerk_config returns ClerkConfig instance."""
        # Clear cache
        get_clerk_config.cache_clear()

        config = get_clerk_config()

        assert isinstance(config, ClerkConfig)

    def test_caches_config(self) -> None:
        """Test get_clerk_config caches result."""
        # Clear cache
        get_clerk_config.cache_clear()

        config1 = get_clerk_config()
        config2 = get_clerk_config()

        assert config1 is config2

    def test_reads_from_environment(self) -> None:
        """Test get_clerk_config reads from environment."""
        # Clear cache
        get_clerk_config.cache_clear()

        env_vars = {
            "CLERK_PUBLISHABLE_KEY": "pk_test_env",
            "CLERK_SECRET_KEY": "sk_test_env",
            "CLERK_ISSUER": "https://env.clerk.example.com",
        }

        with mock.patch.dict(os.environ, env_vars, clear=False):
            get_clerk_config.cache_clear()
            config = get_clerk_config()

            assert config.publishable_key == "pk_test_env"
            assert config.secret_key == "sk_test_env"
            assert config.issuer == "https://env.clerk.example.com"

        # Clear cache after test
        get_clerk_config.cache_clear()
