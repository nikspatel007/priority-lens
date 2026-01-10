"""API route modules."""

from priority_lens.api.routes.connections import router as connections_router
from priority_lens.api.routes.connections import set_connection_service
from priority_lens.api.routes.health import router as health_router
from priority_lens.api.routes.inbox import router as inbox_router
from priority_lens.api.routes.inbox import set_session_factory as set_inbox_session
from priority_lens.api.routes.projects import router as projects_router
from priority_lens.api.routes.projects import set_session_factory as set_projects_session
from priority_lens.api.routes.tasks import router as tasks_router
from priority_lens.api.routes.tasks import set_session_factory as set_tasks_session
from priority_lens.api.routes.webhooks import router as webhooks_router
from priority_lens.api.routes.webhooks import set_push_service

__all__ = [
    "connections_router",
    "health_router",
    "inbox_router",
    "projects_router",
    "set_connection_service",
    "set_inbox_session",
    "set_projects_session",
    "set_push_service",
    "set_tasks_session",
    "tasks_router",
    "webhooks_router",
]
