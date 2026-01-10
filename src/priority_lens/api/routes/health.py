"""Health check endpoints."""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from priority_lens.api.dependencies import DbDep

router = APIRouter(prefix="/health", tags=["health"])
logger = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")


class ReadyResponse(BaseModel):
    """Readiness check response model."""

    status: str = Field(description="Overall readiness status")
    database: str = Field(description="Database connection status")
    checks: dict[str, bool] = Field(description="Individual check results")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check",
    description="Basic health check endpoint that returns API status.",
)
async def health_check() -> HealthResponse:
    """Check if the API is running.

    Returns:
        HealthResponse with status and version.
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
    description="Check if the API is ready to serve requests (including database).",
)
async def readiness_check(db: DbDep) -> ReadyResponse:
    """Check if the API is ready to serve requests.

    This endpoint verifies all dependencies (database, etc.) are available.

    Args:
        db: Database instance from dependency injection.

    Returns:
        ReadyResponse with detailed status of each dependency.
    """
    checks: dict[str, bool] = {}

    # Check database connection
    db_healthy = await db.check_connection()
    checks["database"] = db_healthy

    # Determine overall status
    all_healthy = all(checks.values())
    status = "ready" if all_healthy else "not_ready"
    db_status = "connected" if db_healthy else "disconnected"

    if not all_healthy:
        await logger.awarning(
            "readiness_check_failed",
            checks=checks,
        )

    return ReadyResponse(
        status=status,
        database=db_status,
        checks=checks,
    )
