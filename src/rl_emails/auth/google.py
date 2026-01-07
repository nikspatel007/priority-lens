"""Google OAuth2 implementation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx

from rl_emails.auth.oauth import GoogleTokens, OAuthError


class GoogleOAuth:
    """Google OAuth2 implementation for Gmail API access.

    This class handles the OAuth2 flow for Google authentication:
    1. Generate authorization URL for user consent
    2. Exchange authorization code for access/refresh tokens
    3. Refresh expired access tokens

    Attributes:
        AUTHORIZATION_URL: Google's OAuth2 authorization endpoint.
        TOKEN_URL: Google's OAuth2 token endpoint.
        REVOKE_URL: Google's token revocation endpoint.
        SCOPES: Required OAuth2 scopes for Gmail readonly access.
    """

    AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    REVOKE_URL = "https://oauth2.googleapis.com/revoke"
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        """Initialize Google OAuth client.

        Args:
            client_id: Google OAuth2 client ID.
            client_secret: Google OAuth2 client secret.
            redirect_uri: URI to redirect to after authorization.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self, state: str | None = None) -> str:
        """Generate OAuth authorization URL.

        This URL should be opened in a browser for the user to grant access.

        Args:
            state: Optional state parameter for CSRF protection.

        Returns:
            Full authorization URL with query parameters.
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        if state:
            params["state"] = state

        return f"{self.AUTHORIZATION_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> GoogleTokens:
        """Exchange authorization code for tokens.

        After the user grants access, Google redirects back with an
        authorization code. This method exchanges that code for
        access and refresh tokens.

        Args:
            code: Authorization code from OAuth callback.

        Returns:
            GoogleTokens with access_token, refresh_token, and expiration.

        Raises:
            OAuthError: If the exchange fails (e.g., invalid code).
        """
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.TOKEN_URL, data=data)
            result = response.json()

        if response.status_code != 200:
            error = result.get("error", "unknown_error")
            description = result.get("error_description")
            raise OAuthError(error, description)

        # Calculate expiration time
        expires_in = result.get("expires_in", 3600)
        expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)

        # Parse scopes (space-separated string)
        scope_str = result.get("scope", "")
        scopes = scope_str.split() if scope_str else []

        return GoogleTokens(
            access_token=result["access_token"],
            refresh_token=result["refresh_token"],
            expires_at=expires_at,
            scopes=scopes,
        )

    async def refresh_token(self, refresh_token: str) -> GoogleTokens:
        """Refresh expired access token.

        Use this to get a new access token when the current one expires.
        The refresh token itself typically doesn't expire.

        Args:
            refresh_token: The refresh token from a previous authorization.

        Returns:
            GoogleTokens with new access_token (refresh_token may be same).

        Raises:
            OAuthError: If the refresh fails (e.g., revoked token).
        """
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.TOKEN_URL, data=data)
            result = response.json()

        if response.status_code != 200:
            error = result.get("error", "unknown_error")
            description = result.get("error_description")
            raise OAuthError(error, description)

        # Calculate expiration time
        expires_in = result.get("expires_in", 3600)
        expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)

        # Parse scopes
        scope_str = result.get("scope", "")
        scopes = scope_str.split() if scope_str else []

        # Refresh response may not include new refresh_token
        new_refresh_token = result.get("refresh_token", refresh_token)

        return GoogleTokens(
            access_token=result["access_token"],
            refresh_token=new_refresh_token,
            expires_at=expires_at,
            scopes=scopes,
        )

    async def revoke_token(self, token: str) -> None:
        """Revoke an access or refresh token.

        This invalidates the token and removes the user's consent.
        After revocation, the user must re-authorize.

        Args:
            token: Access token or refresh token to revoke.

        Raises:
            OAuthError: If revocation fails.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.REVOKE_URL,
                params={"token": token},
            )

        if response.status_code != 200:
            # Revoke endpoint returns plain text errors
            raise OAuthError("revocation_failed", response.text)
