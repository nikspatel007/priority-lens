"""API route modules."""

from rl_emails.api.routes.connections import router as connections_router
from rl_emails.api.routes.connections import set_connection_service
from rl_emails.api.routes.health import router as health_router
from rl_emails.api.routes.webhooks import router as webhooks_router
from rl_emails.api.routes.webhooks import set_push_service

__all__ = [
    "connections_router",
    "health_router",
    "set_connection_service",
    "set_push_service",
    "webhooks_router",
]
