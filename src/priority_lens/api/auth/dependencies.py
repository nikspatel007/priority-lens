"""Authentication dependencies for FastAPI."""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import Depends, Header, Request

from priority_lens.api.auth.clerk import ClerkJWTValidator, ClerkUser
from priority_lens.api.auth.config import ClerkConfig, get_clerk_config
from priority_lens.api.auth.exceptions import (
    AuthenticationError,
)

logger = structlog.get_logger(__name__)

# Global validator instance (initialized lazily)
_validator: ClerkJWTValidator | None = None


def get_jwt_validator(
    config: Annotated[ClerkConfig, Depends(get_clerk_config)],
) -> ClerkJWTValidator:
    """Get the JWT validator instance.

    Args:
        config: Clerk configuration.

    Returns:
        ClerkJWTValidator instance.

    Raises:
        AuthenticationError: If Clerk is not configured.
    """
    global _validator

    if not config.is_configured:
        raise AuthenticationError(detail="Authentication not configured")

    if _validator is None:
        _validator = ClerkJWTValidator(config)

    return _validator


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Extract token from Authorization header.

    Args:
        authorization: The Authorization header value.

    Returns:
        The token string or None if not present.
    """
    if not authorization:
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    return parts[1]


async def get_current_user(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    validator: Annotated[ClerkJWTValidator, Depends(get_jwt_validator)] = None,  # type: ignore[assignment]
) -> ClerkUser:
    """Get the current authenticated user from JWT.

    This dependency requires a valid JWT token. Use get_current_user_optional
    for endpoints that can work with or without authentication.

    Args:
        request: The FastAPI request object.
        authorization: Authorization header with Bearer token.
        validator: JWT validator instance.

    Returns:
        ClerkUser extracted from the valid JWT.

    Raises:
        AuthenticationError: If no token provided.
        InvalidTokenError: If token is invalid.
        TokenExpiredError: If token has expired.
    """
    token = _extract_bearer_token(authorization)

    if not token:
        raise AuthenticationError(detail="Authorization header required")

    user = validator.validate_token(token)

    # Store user in request state for logging/middleware
    request.state.user = user

    await logger.ainfo(
        "user_authenticated",
        user_id=user.id,
        email=user.email,
    )

    return user


async def get_current_user_optional(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    config: Annotated[ClerkConfig | None, Depends(get_clerk_config)] = None,
) -> ClerkUser | None:
    """Get the current user if authenticated, None otherwise.

    This dependency does not require authentication. Use for endpoints
    that have different behavior for authenticated vs anonymous users.

    Args:
        request: The FastAPI request object.
        authorization: Authorization header with Bearer token.
        config: Clerk configuration.

    Returns:
        ClerkUser if valid token provided, None otherwise.
    """
    token = _extract_bearer_token(authorization)

    if not token:
        return None

    if config is None or not config.is_configured:
        return None

    try:
        validator = ClerkJWTValidator(config)
        user = validator.validate_token(token)
        request.state.user = user
        return user
    except Exception:
        # Silent failure for optional auth
        return None


async def get_api_key_user(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    config: Annotated[ClerkConfig | None, Depends(get_clerk_config)] = None,
) -> ClerkUser | None:
    """Authenticate via API key for service-to-service calls.

    Args:
        request: The FastAPI request object.
        x_api_key: API key from X-API-Key header.
        config: Clerk configuration with allowed API keys.

    Returns:
        Service user if valid API key, None otherwise.
    """
    if not x_api_key or config is None:
        return None

    if x_api_key not in config.api_keys:
        return None

    # Create a service user for API key authentication
    service_user = ClerkUser(
        id="service",
        email=None,
        first_name="Service",
        last_name="Account",
        metadata={"auth_method": "api_key"},
    )

    request.state.user = service_user

    await logger.ainfo(
        "api_key_authenticated",
        user_id=service_user.id,
    )

    return service_user


async def get_current_user_or_api_key(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    config: Annotated[ClerkConfig | None, Depends(get_clerk_config)] = None,
) -> ClerkUser:
    """Get user from JWT or API key, requiring at least one.

    Tries JWT first, then falls back to API key authentication.

    Args:
        request: The FastAPI request object.
        authorization: Authorization header with Bearer token.
        x_api_key: API key from X-API-Key header.
        config: Clerk configuration.

    Returns:
        ClerkUser from JWT or API key.

    Raises:
        AuthenticationError: If neither method succeeds.
    """
    # Try JWT first
    token = _extract_bearer_token(authorization)
    if token and config and config.is_configured:
        try:
            validator = ClerkJWTValidator(config)
            user = validator.validate_token(token)
            request.state.user = user
            return user
        except Exception:
            pass  # Fall through to API key

    # Try API key
    if x_api_key and config and x_api_key in config.api_keys:
        service_user = ClerkUser(
            id="service",
            email=None,
            first_name="Service",
            last_name="Account",
            metadata={"auth_method": "api_key"},
        )
        request.state.user = service_user
        return service_user

    raise AuthenticationError(detail="Valid JWT or API key required")


# Type aliases for dependency injection
CurrentUser = Annotated[ClerkUser, Depends(get_current_user)]
CurrentUserOptional = Annotated[ClerkUser | None, Depends(get_current_user_optional)]
CurrentUserOrApiKey = Annotated[ClerkUser, Depends(get_current_user_or_api_key)]
