"""Authentication module for Clerk JWT validation."""

from rl_emails.api.auth.clerk import ClerkJWTValidator, ClerkUser
from rl_emails.api.auth.config import ClerkConfig, get_clerk_config
from rl_emails.api.auth.dependencies import (
    CurrentUser,
    CurrentUserOptional,
    CurrentUserOrApiKey,
    get_api_key_user,
    get_current_user,
    get_current_user_optional,
    get_current_user_or_api_key,
)
from rl_emails.api.auth.exceptions import (
    AuthenticationError,
    InsufficientPermissionsError,
    InvalidTokenError,
    TokenExpiredError,
)
from rl_emails.api.auth.middleware import (
    AuthenticationMiddleware,
    add_authentication_middleware,
)
from rl_emails.api.auth.user_sync import UserSyncService

__all__ = [
    # Config
    "ClerkConfig",
    "get_clerk_config",
    # Validator
    "ClerkJWTValidator",
    "ClerkUser",
    # Dependencies
    "get_current_user",
    "get_current_user_optional",
    "get_api_key_user",
    "get_current_user_or_api_key",
    # Type aliases
    "CurrentUser",
    "CurrentUserOptional",
    "CurrentUserOrApiKey",
    # Exceptions
    "AuthenticationError",
    "InvalidTokenError",
    "TokenExpiredError",
    "InsufficientPermissionsError",
    # Middleware
    "AuthenticationMiddleware",
    "add_authentication_middleware",
    # User sync
    "UserSyncService",
]
