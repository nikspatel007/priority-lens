"""Agent API endpoints for cancellation and control."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.api.auth.dependencies import CurrentUserOrApiKey

if TYPE_CHECKING:
    from priority_lens.api.auth.clerk import ClerkUser
    from priority_lens.services.livekit_service import LiveKitService

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/agent", tags=["agent"])
logger = structlog.get_logger(__name__)

# Dependencies - will be overridden in app setup
_session_factory: SessionFactory | None = None
_livekit_service: LiveKitService | None = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session.

    Yields:
        AsyncSession instance.

    Raises:
        HTTPException: If database not configured.
    """
    if _session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured",
        )
    async with _session_factory() as session:
        yield session


def set_session_factory(factory: SessionFactory | None) -> None:
    """Set the session factory.

    Args:
        factory: AsyncSession factory to use.
    """
    global _session_factory
    _session_factory = factory


def set_livekit_service(service: LiveKitService | None) -> None:
    """Set the LiveKit service.

    Args:
        service: LiveKit service to use.
    """
    global _livekit_service
    _livekit_service = service


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _get_user_id(user: ClerkUser) -> UUID:
    """Get UUID from ClerkUser.id string.

    Args:
        user: Authenticated user.

    Returns:
        User ID as UUID.
    """
    try:
        return UUID(user.id)
    except ValueError:
        import hashlib

        hash_bytes = hashlib.md5(user.id.encode()).digest()  # noqa: S324
        return UUID(bytes=hash_bytes)


def _get_org_id(user: ClerkUser) -> UUID:
    """Get organization UUID from ClerkUser.

    Args:
        user: Authenticated user.

    Returns:
        Organization ID as UUID.
    """
    return _get_user_id(user)


# ============================================================================
# Request/Response schemas
# ============================================================================


class CancelRequest(BaseModel):
    """Request to cancel agent execution."""

    correlation_id: UUID = Field(..., description="Correlation ID of the turn to cancel")
    reason: str = Field("user_request", description="Reason for cancellation")


class CancelResponse(BaseModel):
    """Response from cancel request."""

    ok: bool = Field(..., description="Whether cancellation was successful")
    correlation_id: UUID = Field(..., description="Correlation ID that was cancelled")
    message: str = Field(..., description="Result message")


class AgentStatusResponse(BaseModel):
    """Response for agent status check."""

    active_sessions: int = Field(..., description="Number of active streaming sessions")
    is_session_active: bool = Field(..., description="Whether specified session is active")
    correlation_id: UUID | None = Field(None, description="Correlation ID checked")


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/cancel", response_model=CancelResponse)
async def cancel_agent(
    data: CancelRequest,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> CancelResponse:
    """Cancel an active agent execution.

    This endpoint allows clients to cancel an in-progress agent turn,
    useful for barge-in scenarios where the user wants to interrupt.

    The cancellation will:
    1. Mark the streaming session as cancelled
    2. Emit a system.cancel event
    3. Emit a turn.agent.close event with reason "cancelled"
    """
    from priority_lens.services.agent_streaming import (
        AgentStreamingService,
    )

    user_id = _get_user_id(user)

    streaming_service = AgentStreamingService(session, _livekit_service)

    # Try to cancel the agent
    cancelled = await streaming_service.cancel_agent(data.correlation_id)

    if cancelled:
        await logger.ainfo(
            "agent_cancel_requested",
            correlation_id=str(data.correlation_id),
            user_id=str(user_id),
            reason=data.reason,
        )
        return CancelResponse(
            ok=True,
            correlation_id=data.correlation_id,
            message="Agent execution cancelled",
        )
    else:
        # Session not found - may have already completed
        await logger.awarn(
            "agent_cancel_not_found",
            correlation_id=str(data.correlation_id),
            user_id=str(user_id),
        )
        return CancelResponse(
            ok=False,
            correlation_id=data.correlation_id,
            message="No active session found for correlation ID",
        )


@router.get("/status/{correlation_id}", response_model=AgentStatusResponse)
async def get_agent_status(
    correlation_id: UUID,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> AgentStatusResponse:
    """Check if an agent session is active.

    This endpoint allows clients to check the status of an agent
    streaming session.
    """
    from priority_lens.services.agent_streaming import AgentStreamingService

    streaming_service = AgentStreamingService(session, _livekit_service)

    is_active = streaming_service.is_session_active(correlation_id)
    active_count = streaming_service.get_active_session_count()

    return AgentStatusResponse(
        active_sessions=active_count,
        is_session_active=is_active,
        correlation_id=correlation_id,
    )
