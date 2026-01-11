"""Smart Digest API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.api.auth.dependencies import CurrentUserOrApiKey
from priority_lens.schemas.digest import DigestResponse
from priority_lens.services.digest_service import DigestService

if TYPE_CHECKING:
    from priority_lens.api.auth.clerk import ClerkUser

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/digest", tags=["digest"])
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


@router.get("", response_model=DigestResponse)
async def get_smart_digest(
    user: CurrentUserOrApiKey,
    session: SessionDep,
    max_todos: int = Query(5, ge=1, le=20, description="Maximum to-do items"),
    max_topics: int = Query(5, ge=1, le=20, description="Maximum topic items"),
) -> DigestResponse:
    """Get personalized smart digest.

    Returns a personalized daily digest with:
    - Time-appropriate greeting
    - Suggested to-dos (actionable items from emails)
    - Topics to catch up on (grouped conversations)

    The digest is designed for the "AI Inbox" experience, helping users
    quickly understand what needs attention and what's happening.
    """
    service = DigestService(session)
    user_id = _get_user_id(user)

    # Get user's name for personalized greeting
    user_name = user.first_name or user.full_name

    await logger.ainfo(
        "generating_digest",
        user_id=str(user_id),
        max_todos=max_todos,
        max_topics=max_topics,
    )

    return await service.get_digest(
        user_id,
        max_todos=max_todos,
        max_topics=max_topics,
        user_name=user_name,
    )
