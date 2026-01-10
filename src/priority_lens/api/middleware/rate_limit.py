"""Rate limiting middleware using slowapi."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from priority_lens.api.exceptions import ProblemDetail

if TYPE_CHECKING:
    from fastapi import FastAPI

    from priority_lens.api.config import APIConfig

logger = structlog.get_logger(__name__)

# Global limiter instance (configured during setup)
limiter: Limiter | None = None


def get_limiter() -> Limiter:
    """Get the configured rate limiter.

    Returns:
        The global Limiter instance.

    Raises:
        RuntimeError: If limiter has not been configured.
    """
    if limiter is None:
        raise RuntimeError("Rate limiter not configured. Call setup_rate_limit first.")
    return limiter


def _get_request_identifier(request: Request) -> str:
    """Get identifier for rate limiting from request.

    Uses X-User-ID header if present (authenticated users),
    otherwise falls back to IP address.

    Args:
        request: The incoming request.

    Returns:
        Identifier string for rate limiting.
    """
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors with Problem Details.

    Args:
        request: The incoming request.
        exc: The rate limit exceeded exception.

    Returns:
        JSON response with Problem Details format.
    """
    # Parse retry-after from exception if available
    retry_after = None
    if hasattr(exc, "detail") and isinstance(exc.detail, str):
        # slowapi includes retry info in detail string
        # Extract first number from detail string
        parts = exc.detail.split()
        for part in parts:
            if part.isdigit():
                retry_after = int(part)
                break

    problem = ProblemDetail(
        type="/errors/rate-limit",
        title="Too Many Requests",
        status=429,
        detail="Rate limit exceeded. Please slow down your requests.",
        extensions={"retry_after": retry_after} if retry_after else {},
    )

    await logger.awarning(
        "rate_limit_exceeded",
        path=str(request.url.path),
        identifier=_get_request_identifier(request),
        retry_after=retry_after,
    )

    response = JSONResponse(
        status_code=429,
        content=problem.to_dict(),
        media_type="application/problem+json",
    )

    if retry_after:
        response.headers["Retry-After"] = str(retry_after)

    return response


def setup_rate_limit(app: FastAPI, config: APIConfig) -> None:
    """Configure rate limiting for the application.

    Args:
        app: FastAPI application instance.
        config: API configuration with rate limit settings.
    """
    global limiter

    # Create limiter with request identifier function
    limiter = Limiter(
        key_func=_get_request_identifier,
        default_limits=[f"{config.rate_limit_requests}/{config.rate_limit_window}second"],
        enabled=True,
        storage_uri="memory://",  # Use memory storage (can be Redis for distributed)
    )

    # Set limiter on app state
    app.state.limiter = limiter

    # Add rate limit exceeded handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
