"""Actions API endpoints for SDUI action handling."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.api.auth.dependencies import CurrentUserOrApiKey
from priority_lens.services.action_handlers import (
    ActionContext,
    ActionNotFoundError,
    ActionResultStatus,
    ActionService,
    action_registry,
)

if TYPE_CHECKING:
    from priority_lens.api.auth.clerk import ClerkUser

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/actions", tags=["actions"])
logger = structlog.get_logger(__name__)

# Database session dependency - will be overridden in app setup
_session_factory: SessionFactory | None = None


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


class ActionRequest(BaseModel):
    """Request to execute an action."""

    id: str = Field(..., description="Unique action ID for idempotency")
    type: str = Field(..., description="Action type (archive, complete, snooze, etc.)")
    thread_id: UUID = Field(..., description="Thread this action belongs to")
    session_id: UUID | None = Field(None, description="Optional session ID")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific payload",
    )


class ActionResponse(BaseModel):
    """Response from action execution."""

    ok: bool = Field(..., description="Whether action succeeded")
    action_id: str = Field(..., description="The action ID from the request")
    status: str = Field(..., description="Result status (success, failure, pending)")
    message: str = Field(..., description="Human-readable result message")
    data: dict[str, Any] | None = Field(None, description="Action-specific result data")
    error: str | None = Field(None, description="Error code if failed")


class ActionTypesResponse(BaseModel):
    """Response listing available action types."""

    types: list[str] = Field(..., description="List of registered action types")


# ============================================================================
# Endpoints
# ============================================================================


@router.post("", response_model=ActionResponse)
async def execute_action(
    data: ActionRequest,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> ActionResponse:
    """Execute an SDUI action.

    This endpoint handles actions triggered by UI components (buttons, etc.).
    Actions are executed and the result is emitted as a canonical event.

    Supported action types:
    - `archive`: Archive an email
    - `complete`: Complete a task
    - `dismiss`: Dismiss a task
    - `snooze`: Snooze an email or task
    - `navigate`: Navigate to a route
    - `reply`: Reply to an email
    - `delete`: Delete an item
    """
    org_id = _get_org_id(user)
    user_id = _get_user_id(user)

    ctx = ActionContext(
        thread_id=data.thread_id,
        org_id=org_id,
        user_id=user_id,
        session_id=data.session_id,
        action_id=data.id,
    )

    action_service = ActionService(session)

    try:
        result = await action_service.execute_action(
            ctx=ctx,
            action_type=data.type,
            payload=data.payload,
        )

        return ActionResponse(
            ok=result.status == ActionResultStatus.SUCCESS,
            action_id=data.id,
            status=result.status.value,
            message=result.message,
            data=result.data,
            error=result.error,
        )

    except ActionNotFoundError as e:
        await logger.awarn(
            "action_type_not_found",
            action_type=data.type,
            user_id=str(user_id),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action type: {e.action_type}",
        ) from e


@router.get("/types", response_model=ActionTypesResponse)
async def list_action_types(
    user: CurrentUserOrApiKey,
) -> ActionTypesResponse:
    """List available action types.

    Returns the list of registered action handlers.
    """
    return ActionTypesResponse(types=action_registry.registered_types)
