"""Structured logging middleware with request context."""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.context import correlation_id
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from fastapi import FastAPI

    from priority_lens.api.config import APIConfig

# Context variables for request-scoped logging
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)

logger = structlog.get_logger(__name__)


def add_correlation_id(
    _logger: structlog.types.WrappedLogger,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Add correlation ID to log events.

    Args:
        _logger: The wrapped logger (unused).
        _method_name: The logging method name (unused).
        event_dict: The event dictionary to modify.

    Returns:
        Modified event dictionary with correlation_id.
    """
    cid = correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid

    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id

    user_id = user_id_var.get()
    if user_id:
        event_dict["user_id"] = user_id

    return event_dict


def configure_structlog(json_format: bool = True, log_level: str = "INFO") -> None:
    """Configure structlog for the application.

    Args:
        json_format: Whether to output logs as JSON.
        log_level: The logging level to use.
    """
    import logging

    # Set up standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Configure processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and log timing information.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler.

        Returns:
            The response from the handler.
        """
        # Generate and store request ID
        request_id = str(uuid4())
        request_id_var.set(request_id)

        # Extract user ID from headers if present
        user_id = request.headers.get("X-User-ID")
        if user_id:
            user_id_var.set(user_id)

        start_time = time.perf_counter()

        # Log request
        await logger.ainfo(
            "request_started",
            method=request.method,
            path=str(request.url.path),
            query=str(request.url.query) if request.url.query else None,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            await logger.ainfo(
                "request_completed",
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            await logger.aerror(
                "request_failed",
                method=request.method,
                path=str(request.url.path),
                duration_ms=round(duration_ms, 2),
                error=str(exc),
            )
            raise

        finally:
            # Clean up context vars
            request_id_var.set(None)
            user_id_var.set(None)


def setup_logging(app: FastAPI, config: APIConfig) -> None:
    """Set up logging middleware and configuration.

    Args:
        app: FastAPI application instance.
        config: API configuration with logging settings.
    """
    # Configure structlog
    configure_structlog(
        json_format=config.log_json,
        log_level=config.log_level,
    )

    # Add correlation ID middleware (generates ID if not present)
    app.add_middleware(
        CorrelationIdMiddleware,
        header_name="X-Correlation-ID",
        generator=lambda: str(uuid4()),
        transformer=lambda x: x,
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
