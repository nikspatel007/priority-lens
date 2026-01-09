"""Authentication middleware for FastAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rl_emails.api.auth.clerk import ClerkJWTValidator
from rl_emails.api.auth.config import get_clerk_config

if TYPE_CHECKING:
    from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware that attempts to authenticate requests.

    This middleware:
    - Extracts JWT from Authorization header if present
    - Validates the token and sets user on request.state
    - Does NOT block requests - use dependencies for that
    - Logs authentication attempts for monitoring

    Use this middleware for:
    - Adding user context to all requests for logging
    - Preemptively validating tokens before route handlers
    - Supporting both authenticated and anonymous requests

    For blocking unauthenticated requests, use the get_current_user
    dependency instead.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the authentication middleware.

        Args:
            app: The ASGI application to wrap.
        """
        super().__init__(app)
        self._validator: ClerkJWTValidator | None = None

    @property
    def validator(self) -> ClerkJWTValidator | None:
        """Get or create the JWT validator (lazy initialization).

        Returns None if Clerk is not configured.
        """
        if self._validator is None:
            config = get_clerk_config()
            if config.is_configured:
                self._validator = ClerkJWTValidator(config)
        return self._validator

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request and attempt authentication.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            The response from downstream handlers.
        """
        # Initialize user state to None
        request.state.user = None

        # Try to authenticate if Authorization header present
        auth_header = request.headers.get("Authorization")
        if auth_header and self.validator is not None:
            await self._try_authenticate(request, auth_header)

        return await call_next(request)

    async def _try_authenticate(
        self,
        request: Request,
        auth_header: str,
    ) -> None:
        """Attempt to authenticate the request.

        This method does not raise exceptions - it silently fails
        and leaves request.state.user as None.

        Args:
            request: The incoming request.
            auth_header: The Authorization header value.
        """
        # Extract bearer token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return

        token = parts[1]

        try:
            # validator is guaranteed non-None by caller (dispatch checks first)
            user = self.validator.validate_token(token)  # type: ignore[union-attr]
            request.state.user = user

            await logger.ainfo(
                "middleware_auth_success",
                user_id=user.id,
                path=str(request.url.path),
            )

        except Exception as e:
            # Log but don't block - let route handlers decide
            await logger.awarning(
                "middleware_auth_failed",
                error=str(e),
                path=str(request.url.path),
            )


def add_authentication_middleware(app: ASGIApp) -> ASGIApp:
    """Add authentication middleware to an ASGI app.

    This is a factory function for adding the middleware.

    Args:
        app: The ASGI application to wrap.

    Returns:
        The wrapped application with authentication middleware.
    """
    return AuthenticationMiddleware(app)
