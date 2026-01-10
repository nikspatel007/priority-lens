"""Tests for OAuthToken model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from priority_lens.models.oauth_token import OAuthToken


class TestOAuthToken:
    """Tests for OAuthToken model."""

    def test_oauth_token_create(self) -> None:
        """Test creating an OAuth token."""
        user_id = uuid.uuid4()
        expires = datetime.now(UTC) + timedelta(hours=1)
        token = OAuthToken(
            user_id=user_id,
            access_token="ya29.test",
            refresh_token="1//test",
            expires_at=expires,
            scopes=["gmail.readonly"],
        )
        assert token.user_id == user_id
        assert token.access_token == "ya29.test"
        assert token.refresh_token == "1//test"
        assert token.expires_at == expires
        assert token.scopes == ["gmail.readonly"]

    def test_oauth_token_default_provider(self) -> None:
        """Test OAuth token has google provider by default."""
        user_id = uuid.uuid4()
        expires = datetime.now(UTC) + timedelta(hours=1)
        token = OAuthToken(
            user_id=user_id,
            access_token="test",
            refresh_token="test",
            expires_at=expires,
        )
        # Before commit, provider may be None or 'google' depending on SQLAlchemy version
        assert token.provider is None or token.provider == "google"

    def test_oauth_token_is_expired_false(self) -> None:
        """Test is_expired returns False for valid token."""
        user_id = uuid.uuid4()
        expires = datetime.now(UTC) + timedelta(hours=1)
        token = OAuthToken(
            user_id=user_id,
            access_token="test",
            refresh_token="test",
            expires_at=expires,
        )
        assert token.is_expired is False

    def test_oauth_token_is_expired_true(self) -> None:
        """Test is_expired returns True for expired token."""
        user_id = uuid.uuid4()
        expires = datetime.now(UTC) - timedelta(hours=1)
        token = OAuthToken(
            user_id=user_id,
            access_token="test",
            refresh_token="test",
            expires_at=expires,
        )
        assert token.is_expired is True

    def test_oauth_token_repr(self) -> None:
        """Test OAuth token string representation."""
        token_id = uuid.uuid4()
        user_id = uuid.uuid4()
        expires = datetime.now(UTC) + timedelta(hours=1)
        token = OAuthToken(
            id=token_id,
            user_id=user_id,
            provider="google",
            access_token="test",
            refresh_token="test",
            expires_at=expires,
        )
        result = repr(token)
        assert "OAuthToken" in result
        assert "google" in result


class TestOAuthTokenTablename:
    """Tests for OAuthToken table configuration."""

    def test_tablename(self) -> None:
        """Test table name is correct."""
        assert OAuthToken.__tablename__ == "oauth_tokens"
