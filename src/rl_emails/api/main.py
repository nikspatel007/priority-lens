"""FastAPI application factory and lifespan management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from rl_emails.api.config import APIConfig, get_api_config
from rl_emails.api.database import Database, set_database
from rl_emails.api.middleware import (
    setup_cors,
    setup_error_handlers,
    setup_logging,
    setup_rate_limit,
)
from rl_emails.api.routes import (
    connections_router,
    health_router,
    webhooks_router,
)

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        Control to the application after startup is complete.
    """
    config: APIConfig = app.state.config

    # Startup
    await logger.ainfo(
        "application_starting",
        environment=config.environment,
        host=config.host,
        port=config.port,
    )

    # Initialize database
    db = Database(config)
    await db.connect()
    set_database(db)
    app.state.db = db

    await logger.ainfo("database_connected")

    yield

    # Shutdown
    await logger.ainfo("application_shutting_down")

    # Close database connections
    if hasattr(app.state, "db") and app.state.db is not None:
        await app.state.db.disconnect()
        set_database(None)

    await logger.ainfo("application_shutdown_complete")


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional API configuration. If not provided,
                configuration is loaded from environment variables.

    Returns:
        Configured FastAPI application instance.
    """
    if config is None:
        config = get_api_config()

    app = FastAPI(
        title="rl-emails API",
        description="Email ML pipeline API for analyzing Gmail and predicting email priority/actions",
        version="2.0.0",
        docs_url="/docs" if config.is_development else None,
        redoc_url="/redoc" if config.is_development else None,
        openapi_url="/openapi.json" if config.is_development else None,
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config

    # Setup middleware (order matters: last added = first executed)
    # Order: CORS -> Correlation ID -> Request Logging -> Error Handler

    # Error handlers (added first, executes last in middleware chain)
    setup_error_handlers(app)

    # Rate limiting
    setup_rate_limit(app, config)

    # Logging (includes correlation ID middleware)
    setup_logging(app, config)

    # CORS (added last, executes first)
    setup_cors(app, config)

    # Register routes
    app.include_router(health_router)
    app.include_router(connections_router)
    app.include_router(webhooks_router)

    return app


def run_server(config: APIConfig | None = None) -> None:
    """Run the API server using uvicorn.

    Args:
        config: Optional API configuration. If not provided,
                configuration is loaded from environment variables.
    """
    import uvicorn

    if config is None:
        config = get_api_config()

    uvicorn.run(
        "rl_emails.api.main:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        reload=config.is_development,
        log_level=config.log_level.lower(),
    )
