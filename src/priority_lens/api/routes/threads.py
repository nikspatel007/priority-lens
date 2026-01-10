"""Thread API endpoints for Voice AI conversations."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.api.auth.dependencies import CurrentUserOrApiKey
from priority_lens.repositories.conversation_thread import ThreadRepository
from priority_lens.repositories.event import EventRepository
from priority_lens.repositories.session import SessionRepository
from priority_lens.schemas.conversation_thread import (
    ThreadCreate,
    ThreadListResponse,
    ThreadResponse,
    ThreadUpdate,
)
from priority_lens.schemas.event import EventListResponse, EventResponse
from priority_lens.schemas.session import (
    SessionCreate,
    SessionListResponse,
    SessionResponse,
)
from priority_lens.schemas.turn import TurnCreate, TurnResponse

if TYPE_CHECKING:
    from priority_lens.api.auth.clerk import ClerkUser

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/threads", tags=["threads"])
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
    # For now, use user_id as org_id (single-user organizations)
    # In future, this could be extracted from user metadata or a separate lookup
    return _get_user_id(user)


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    data: ThreadCreate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> ThreadResponse:
    """Create a new conversation thread."""
    repo = ThreadRepository(session)
    org_id = _get_org_id(user)
    user_id = _get_user_id(user)

    thread = await repo.create(data, org_id, user_id)

    await logger.ainfo(
        "thread_created",
        thread_id=str(thread.id),
        user_id=str(user_id),
    )

    return ThreadResponse.from_orm_with_metadata(thread)


@router.get("", response_model=ThreadListResponse)
async def list_threads(
    user: CurrentUserOrApiKey,
    session: SessionDep,
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> ThreadListResponse:
    """List conversation threads for the authenticated user."""
    repo = ThreadRepository(session)
    org_id = _get_org_id(user)
    user_id = _get_user_id(user)

    threads = await repo.list_by_user(user_id, org_id, limit=limit, offset=offset)
    total = await repo.count_by_user(user_id, org_id)

    return ThreadListResponse(
        threads=[ThreadResponse.from_orm_with_metadata(t) for t in threads],
        total=total,
    )


@router.get("/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: UUID,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> ThreadResponse:
    """Get a specific conversation thread."""
    repo = ThreadRepository(session)
    org_id = _get_org_id(user)

    thread = await repo.get_by_id_and_org(thread_id, org_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    return ThreadResponse.from_orm_with_metadata(thread)


@router.patch("/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: UUID,
    data: ThreadUpdate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> ThreadResponse:
    """Update a conversation thread."""
    repo = ThreadRepository(session)
    org_id = _get_org_id(user)

    thread = await repo.update(thread_id, org_id, data)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    return ThreadResponse.from_orm_with_metadata(thread)


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: UUID,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> None:
    """Delete a conversation thread."""
    repo = ThreadRepository(session)
    org_id = _get_org_id(user)

    deleted = await repo.delete(thread_id, org_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )


# ============================================================================
# Events endpoints
# ============================================================================


@router.get("/{thread_id}/events", response_model=EventListResponse)
async def get_events(
    thread_id: UUID,
    user: CurrentUserOrApiKey,
    session: SessionDep,
    after_seq: int = Query(0, ge=0, description="Fetch events after this sequence number"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
) -> EventListResponse:
    """Fetch events after a sequence number (for reconnection).

    This endpoint supports reconnection by allowing clients to request
    all events after a specific sequence number.
    """
    # First verify thread access
    thread_repo = ThreadRepository(session)
    org_id = _get_org_id(user)

    thread = await thread_repo.get_by_id_and_org(thread_id, org_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    # Get events
    event_repo = EventRepository(session)
    events = await event_repo.get_events_after_seq(thread_id, org_id, after_seq, limit + 1)

    # Check if there are more events
    has_more = len(events) > limit
    if has_more:
        events = events[:limit]

    # Calculate next_seq
    next_seq = events[-1].seq if events else after_seq

    return EventListResponse(
        events=[EventResponse.model_validate(e) for e in events],
        next_seq=next_seq,
        has_more=has_more,
    )


# ============================================================================
# Sessions endpoints
# ============================================================================


@router.post(
    "/{thread_id}/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    thread_id: UUID,
    data: SessionCreate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> SessionResponse:
    """Create a new session for a thread."""
    # First verify thread access
    thread_repo = ThreadRepository(session)
    org_id = _get_org_id(user)

    thread = await thread_repo.get_by_id_and_org(thread_id, org_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    # Create session
    session_repo = SessionRepository(session)
    sess = await session_repo.create(data, thread_id, org_id)

    await logger.ainfo(
        "session_created",
        session_id=str(sess.id),
        thread_id=str(thread_id),
        mode=data.mode.value,
    )

    return SessionResponse.from_orm_with_metadata(sess)


@router.get("/{thread_id}/sessions", response_model=SessionListResponse)
async def list_sessions(
    thread_id: UUID,
    user: CurrentUserOrApiKey,
    session: SessionDep,
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> SessionListResponse:
    """List sessions for a thread."""
    # First verify thread access
    thread_repo = ThreadRepository(session)
    org_id = _get_org_id(user)

    thread = await thread_repo.get_by_id_and_org(thread_id, org_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    # Get sessions
    session_repo = SessionRepository(session)
    sessions = await session_repo.list_by_thread(thread_id, org_id, limit=limit, offset=offset)
    total = await session_repo.count_by_thread(thread_id, org_id)

    return SessionListResponse(
        sessions=[SessionResponse.from_orm_with_metadata(s) for s in sessions],
        total=total,
    )


# ============================================================================
# Turns endpoints
# ============================================================================


@router.post(
    "/{thread_id}/turns",
    response_model=TurnResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_turn(
    thread_id: UUID,
    data: TurnCreate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> TurnResponse:
    """Submit a conversation turn.

    This endpoint accepts text or voice transcript input and creates
    a sequence of events:
    1. turn.user.open - Turn started
    2. ui.text.submit or stt.final - Content event
    3. turn.user.close - Turn completed

    The agent will be invoked asynchronously to respond.
    """
    # Import here to avoid circular imports
    from priority_lens.services.turn_service import TurnService

    # Verify thread access
    thread_repo = ThreadRepository(session)
    org_id = _get_org_id(user)
    user_id = _get_user_id(user)

    thread = await thread_repo.get_by_id_and_org(thread_id, org_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )

    # Submit turn
    turn_service = TurnService(session)
    result = await turn_service.submit_turn(
        thread_id=thread_id,
        org_id=org_id,
        user_id=user_id,
        turn_data=data,
    )

    return result
