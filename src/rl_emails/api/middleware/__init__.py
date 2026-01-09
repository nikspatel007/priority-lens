"""API middleware modules."""

from rl_emails.api.middleware.cors import setup_cors
from rl_emails.api.middleware.error_handler import setup_error_handlers
from rl_emails.api.middleware.logging import setup_logging
from rl_emails.api.middleware.rate_limit import setup_rate_limit

__all__ = [
    "setup_cors",
    "setup_error_handlers",
    "setup_logging",
    "setup_rate_limit",
]
