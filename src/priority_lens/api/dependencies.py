"""FastAPI dependency injection providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.api.database import Database, get_database


def get_db() -> Database:
    """Get the database instance.

    Returns:
        The global Database instance.
    """
    return get_database()


async def get_db_session(
    db: Annotated[Database, Depends(get_db)],
) -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session.

    Yields:
        AsyncSession that is automatically closed on exit.

    Args:
        db: Database instance from dependency injection.
    """
    async with db.session() as session:
        yield session


def get_user_id_optional(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> str | None:
    """Extract optional user ID from request headers.

    Args:
        x_user_id: User ID from X-User-ID header.

    Returns:
        User ID string or None if not provided.
    """
    return x_user_id


def get_org_id_optional(
    x_org_id: Annotated[str | None, Header(alias="X-Org-ID")] = None,
) -> str | None:
    """Extract optional organization ID from request headers.

    Args:
        x_org_id: Organization ID from X-Org-ID header.

    Returns:
        Organization ID string or None if not provided.
    """
    return x_org_id


# Type aliases for dependency injection
DbDep = Annotated[Database, Depends(get_db)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
UserIdDep = Annotated[str | None, Depends(get_user_id_optional)]
OrgIdDep = Annotated[str | None, Depends(get_org_id_optional)]
