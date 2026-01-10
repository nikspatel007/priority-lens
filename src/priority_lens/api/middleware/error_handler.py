"""Global error handler middleware."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from priority_lens.api.exceptions import APIError, ProblemDetail

logger = structlog.get_logger(__name__)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors with Problem Details response.

    Args:
        request: The incoming request.
        exc: The API error that was raised.

    Returns:
        JSON response with Problem Details format.
    """
    await logger.awarning(
        "api_error",
        error_type=exc.problem.type,
        status=exc.problem.status,
        detail=exc.problem.detail,
        path=str(request.url.path),
    )

    return JSONResponse(
        status_code=exc.problem.status,
        content=exc.problem.to_dict(),
        media_type="application/problem+json",
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette HTTP exceptions with Problem Details response.

    Args:
        request: The incoming request.
        exc: The HTTP exception that was raised.

    Returns:
        JSON response with Problem Details format.
    """
    problem = ProblemDetail(
        type="/errors/http",
        title=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
        status=exc.status_code,
        detail=exc.detail if isinstance(exc.detail, str) else None,
    )

    await logger.awarning(
        "http_error",
        status=exc.status_code,
        detail=exc.detail,
        path=str(request.url.path),
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=problem.to_dict(),
        media_type="application/problem+json",
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors with Problem Details response.

    Args:
        request: The incoming request.
        exc: The validation error that was raised.

    Returns:
        JSON response with Problem Details format including validation errors.
    """
    errors = []
    for error in exc.errors():
        loc = error.get("loc", ())
        field = ".".join(str(x) for x in loc) if loc else "unknown"
        errors.append(
            {
                "field": field,
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown"),
            }
        )

    problem = ProblemDetail(
        type="/errors/validation",
        title="Validation Error",
        status=422,
        detail="Request validation failed",
        extensions={"errors": errors},
    )

    await logger.awarning(
        "validation_error",
        error_count=len(errors),
        errors=errors,
        path=str(request.url.path),
    )

    return JSONResponse(
        status_code=422,
        content=problem.to_dict(),
        media_type="application/problem+json",
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with Problem Details response.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        JSON response with generic error Problem Details.
    """
    problem = ProblemDetail(
        type="/errors/internal",
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred",
    )

    await logger.aexception(
        "unhandled_exception",
        exc_type=type(exc).__name__,
        path=str(request.url.path),
    )

    return JSONResponse(
        status_code=500,
        content=problem.to_dict(),
        media_type="application/problem+json",
    )


def setup_error_handlers(app: FastAPI) -> None:
    """Register all error handlers for the application.

    Args:
        app: FastAPI application instance.
    """
    app.add_exception_handler(APIError, api_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
