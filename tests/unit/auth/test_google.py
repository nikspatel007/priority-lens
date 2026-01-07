"""Tests for Google OAuth implementation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.auth.google import GoogleOAuth
from rl_emails.auth.oauth import OAuthError


class TestGoogleOAuthInit:
    """Tests for GoogleOAuth initialization."""

    def test_init_stores_credentials(self) -> None:
        """GoogleOAuth stores client credentials."""
        oauth = GoogleOAuth(
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback",
        )
        assert oauth.client_id == "test-client-id"
        assert oauth.client_secret == "test-client-secret"
        assert oauth.redirect_uri == "http://localhost:8000/callback"

    def test_scopes_include_gmail_readonly(self) -> None:
        """GoogleOAuth includes gmail.readonly scope."""
        assert "https://www.googleapis.com/auth/gmail.readonly" in GoogleOAuth.SCOPES


class TestGetAuthorizationUrl:
    """Tests for get_authorization_url method."""

    def test_includes_client_id(self) -> None:
        """Authorization URL includes client_id."""
        oauth = GoogleOAuth(
            client_id="my-client-id",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )
        url = oauth.get_authorization_url()
        assert "client_id=my-client-id" in url

    def test_includes_redirect_uri(self) -> None:
        """Authorization URL includes redirect_uri."""
        oauth = GoogleOAuth(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost:8000/auth/callback",
        )
        url = oauth.get_authorization_url()
        # URL encoded
        assert "redirect_uri=http" in url
        assert "callback" in url

    def test_includes_scopes(self) -> None:
        """Authorization URL includes required scopes."""
        oauth = GoogleOAuth(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )
        url = oauth.get_authorization_url()
        assert "gmail.readonly" in url

    def test_includes_state_when_provided(self) -> None:
        """Authorization URL includes state parameter when provided."""
        oauth = GoogleOAuth(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )
        url = oauth.get_authorization_url(state="random-state-123")
        assert "state=random-state-123" in url

    def test_excludes_state_when_none(self) -> None:
        """Authorization URL excludes state parameter when not provided."""
        oauth = GoogleOAuth(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )
        url = oauth.get_authorization_url()
        assert "state=" not in url

    def test_includes_offline_access(self) -> None:
        """Authorization URL requests offline access for refresh token."""
        oauth = GoogleOAuth(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )
        url = oauth.get_authorization_url()
        assert "access_type=offline" in url

    def test_includes_prompt_consent(self) -> None:
        """Authorization URL forces consent prompt."""
        oauth = GoogleOAuth(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )
        url = oauth.get_authorization_url()
        assert "prompt=consent" in url


class TestExchangeCode:
    """Tests for exchange_code method."""

    @pytest.mark.asyncio
    async def test_returns_tokens_on_success(self) -> None:
        """exchange_code returns GoogleTokens on success."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_in": 3600,
            "scope": "https://www.googleapis.com/auth/gmail.readonly",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            tokens = await oauth.exchange_code("auth-code-xyz")

        assert tokens.access_token == "access-123"
        assert tokens.refresh_token == "refresh-456"
        assert "gmail.readonly" in tokens.scopes[0]
        assert not tokens.is_expired()

    @pytest.mark.asyncio
    async def test_raises_oauth_error_on_invalid_grant(self) -> None:
        """exchange_code raises OAuthError on invalid grant."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Code has expired or been used",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            with pytest.raises(OAuthError) as exc_info:
                await oauth.exchange_code("expired-code")

        assert exc_info.value.error == "invalid_grant"
        assert "expired" in exc_info.value.description.lower()

    @pytest.mark.asyncio
    async def test_calculates_expiration_time(self) -> None:
        """exchange_code calculates correct expiration time."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_in": 1800,  # 30 minutes
            "scope": "",
        }

        before = datetime.now(UTC)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            tokens = await oauth.exchange_code("code")

        after = datetime.now(UTC)

        # Should expire in approximately 30 minutes
        expected_min = before + timedelta(seconds=1790)
        expected_max = after + timedelta(seconds=1810)
        assert expected_min <= tokens.expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_handles_empty_scope(self) -> None:
        """exchange_code handles empty scope string."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_in": 3600,
            "scope": "",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            tokens = await oauth.exchange_code("code")

        assert tokens.scopes == []


class TestRefreshToken:
    """Tests for refresh_token method."""

    @pytest.mark.asyncio
    async def test_returns_new_access_token(self) -> None:
        """refresh_token returns new access token."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
            "scope": "https://www.googleapis.com/auth/gmail.readonly",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            tokens = await oauth.refresh_token("old-refresh-token")

        assert tokens.access_token == "new-access-token"
        assert not tokens.is_expired()

    @pytest.mark.asyncio
    async def test_preserves_refresh_token_when_not_returned(self) -> None:
        """refresh_token preserves original refresh token if not in response."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
            "scope": "",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            tokens = await oauth.refresh_token("original-refresh-token")

        assert tokens.refresh_token == "original-refresh-token"

    @pytest.mark.asyncio
    async def test_uses_new_refresh_token_when_returned(self) -> None:
        """refresh_token uses new refresh token if provided."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
            "scope": "",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            tokens = await oauth.refresh_token("original-refresh-token")

        assert tokens.refresh_token == "new-refresh-token"

    @pytest.mark.asyncio
    async def test_raises_oauth_error_on_invalid_token(self) -> None:
        """refresh_token raises OAuthError on invalid/revoked token."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Token has been revoked",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            with pytest.raises(OAuthError) as exc_info:
                await oauth.refresh_token("revoked-token")

        assert exc_info.value.error == "invalid_grant"


class TestRevokeToken:
    """Tests for revoke_token method."""

    @pytest.mark.asyncio
    async def test_revokes_token_successfully(self) -> None:
        """revoke_token succeeds with valid token."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            # Should not raise
            await oauth.revoke_token("token-to-revoke")

    @pytest.mark.asyncio
    async def test_raises_oauth_error_on_failure(self) -> None:
        """revoke_token raises OAuthError on failure."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Invalid token"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock()

            with pytest.raises(OAuthError) as exc_info:
                await oauth.revoke_token("invalid-token")

        assert exc_info.value.error == "revocation_failed"
