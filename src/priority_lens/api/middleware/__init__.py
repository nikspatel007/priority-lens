"""API middleware modules."""

from priority_lens.api.middleware.cors import setup_cors
from priority_lens.api.middleware.error_handler import setup_error_handlers
from priority_lens.api.middleware.logging import setup_logging
from priority_lens.api.middleware.rate_limit import setup_rate_limit

__all__ = [
    "setup_cors",
    "setup_error_handlers",
    "setup_logging",
    "setup_rate_limit",
]
