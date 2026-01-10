"""Clerk JWT validation and user extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jwt
import structlog
from jwt import PyJWKClient  # type: ignore[attr-defined]

from priority_lens.api.auth.config import ClerkConfig
from priority_lens.api.auth.exceptions import InvalidTokenError, TokenExpiredError

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ClerkUser:
    """User information extracted from Clerk JWT.

    Attributes:
        id: Clerk user ID (sub claim).
        email: User's primary email address.
        first_name: User's first name.
        last_name: User's last name.
        image_url: URL to user's profile image.
        metadata: Additional user metadata from JWT.
    """

    id: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    image_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str | None:
        """Get user's full name if available."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name


class ClerkJWTValidator:
    """Validates Clerk JWT tokens using JWKS.

    This validator:
    - Fetches public keys from Clerk's JWKS endpoint
    - Validates RS256 signatures
    - Checks token expiration with configurable leeway
    - Extracts user information from claims
    """

    def __init__(self, config: ClerkConfig) -> None:
        """Initialize the JWT validator.

        Args:
            config: Clerk configuration with JWKS URL and validation settings.

        Raises:
            ValueError: If config is missing required fields.
        """
        if not config.effective_jwks_url:
            raise ValueError("JWKS URL is required for JWT validation")

        self._config = config
        self._jwks_client: PyJWKClient | None = None

    @property
    def jwks_client(self) -> PyJWKClient:
        """Get or create the JWKS client (lazy initialization)."""
        if self._jwks_client is None:
            self._jwks_client = PyJWKClient(
                self._config.effective_jwks_url,
                cache_keys=True,
                lifespan=3600,  # Cache keys for 1 hour
            )
        return self._jwks_client

    def validate_token(self, token: str) -> ClerkUser:
        """Validate a JWT token and extract user information.

        Args:
            token: The JWT token string (without Bearer prefix).

        Returns:
            ClerkUser with extracted user information.

        Raises:
            InvalidTokenError: If the token is malformed or signature invalid.
            TokenExpiredError: If the token has expired.
        """
        try:
            # Get the signing key from JWKS
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)

            # Build decode options
            options: dict[str, Any] = {
                "require": ["exp", "sub", "iat"],
                "leeway": self._config.token_leeway_seconds,
            }

            # Build validation kwargs
            decode_kwargs: dict[str, Any] = {
                "algorithms": ["RS256"],
            }

            if self._config.issuer:
                decode_kwargs["issuer"] = self._config.issuer

            if self._config.audience:
                decode_kwargs["audience"] = self._config.audience

            # Decode and validate the token
            payload = jwt.decode(
                token,
                signing_key.key,
                options=options,
                **decode_kwargs,
            )

            return self._extract_user(payload)

        except jwt.ExpiredSignatureError as e:
            logger.warning("token_expired", error=str(e))
            raise TokenExpiredError() from e

        except jwt.InvalidTokenError as e:
            logger.warning("invalid_token", error=str(e))
            raise InvalidTokenError(detail=str(e)) from e

        except Exception as e:
            logger.error("token_validation_failed", error=str(e), exc_info=True)
            raise InvalidTokenError(detail="Token validation failed") from e

    def _extract_user(self, payload: dict[str, Any]) -> ClerkUser:
        """Extract user information from JWT payload.

        Args:
            payload: Decoded JWT payload.

        Returns:
            ClerkUser with extracted information.
        """
        # Extract standard Clerk claims
        user_id = payload["sub"]

        # Clerk can include user info in various locations
        # Check for nested user data first
        user_data = payload.get("user", {})

        email = (
            payload.get("email") or user_data.get("primary_email_address") or user_data.get("email")
        )

        first_name = payload.get("first_name") or user_data.get("first_name")
        last_name = payload.get("last_name") or user_data.get("last_name")
        image_url = payload.get("image_url") or user_data.get("image_url")

        # Collect any additional metadata
        metadata = {}
        if "public_metadata" in payload:
            metadata["public"] = payload["public_metadata"]
        if "private_metadata" in payload:
            metadata["private"] = payload["private_metadata"]
        if "unsafe_metadata" in payload:
            metadata["unsafe"] = payload["unsafe_metadata"]

        return ClerkUser(
            id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            image_url=image_url,
            metadata=metadata if metadata else {},
        )

    def decode_without_verification(self, token: str) -> dict[str, Any]:
        """Decode a token without verification (for debugging).

        WARNING: This should only be used for debugging/logging.
        Never trust the contents of an unverified token.

        Args:
            token: The JWT token string.

        Returns:
            Decoded payload without signature verification.
        """
        return jwt.decode(token, options={"verify_signature": False})
