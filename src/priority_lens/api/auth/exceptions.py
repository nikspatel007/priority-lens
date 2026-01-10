"""Authentication exceptions."""

from __future__ import annotations

from priority_lens.api.exceptions import APIError


class AuthenticationError(APIError):
    """Base authentication error (401)."""

    def __init__(self, detail: str = "Authentication failed") -> None:
        """Initialize authentication error.

        Args:
            detail: Description of the authentication failure.
        """
        super().__init__(
            title="Unauthorized",
            detail=detail,
            status=401,
            error_type="/errors/authentication",
        )


class InvalidTokenError(AuthenticationError):
    """Invalid or malformed token error."""

    def __init__(self, detail: str = "Invalid or malformed token") -> None:
        """Initialize invalid token error.

        Args:
            detail: Description of why the token is invalid.
        """
        super().__init__(detail=detail)
        self.problem = self.problem.__class__(
            type="/errors/invalid-token",
            title="Invalid Token",
            status=401,
            detail=detail,
        )


class TokenExpiredError(AuthenticationError):
    """Token has expired error."""

    def __init__(self, detail: str = "Token has expired") -> None:
        """Initialize token expired error.

        Args:
            detail: Description of the expiration.
        """
        super().__init__(detail=detail)
        self.problem = self.problem.__class__(
            type="/errors/token-expired",
            title="Token Expired",
            status=401,
            detail=detail,
        )


class InsufficientPermissionsError(APIError):
    """User lacks required permissions (403)."""

    def __init__(self, detail: str = "Insufficient permissions") -> None:
        """Initialize insufficient permissions error.

        Args:
            detail: Description of the required permissions.
        """
        super().__init__(
            title="Forbidden",
            detail=detail,
            status=403,
            error_type="/errors/insufficient-permissions",
        )
