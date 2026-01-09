"""User synchronization service for Clerk users."""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.api.auth.clerk import ClerkUser
from rl_emails.models.org_user import OrgUser
from rl_emails.repositories.org_user import OrgUserRepository
from rl_emails.schemas.org_user import OrgUserCreate, OrgUserUpdate

logger = structlog.get_logger(__name__)


class UserSyncService:
    """Service for syncing Clerk users to local database.

    This service:
    - Finds or creates local user records for Clerk users
    - Updates user metadata when it changes
    - Links Clerk users to organizations

    Usage:
        async with get_db_session() as session:
            sync_service = UserSyncService(session)
            local_user = await sync_service.sync_user(clerk_user, org_id)
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the user sync service.

        Args:
            session: Async database session.
        """
        self._session = session
        self._repo = OrgUserRepository(session)

    async def sync_user(
        self,
        clerk_user: ClerkUser,
        org_id: UUID,
    ) -> OrgUser:
        """Sync a Clerk user to the local database.

        Finds an existing user by email or creates a new one.
        Updates user metadata if it has changed.

        Args:
            clerk_user: User information from Clerk JWT.
            org_id: Organization to associate the user with.

        Returns:
            The local OrgUser record.

        Raises:
            ValueError: If clerk_user.email is None.
        """
        if not clerk_user.email:
            raise ValueError("Clerk user must have an email for sync")

        # Try to find existing user by email
        existing_user = await self._repo.get_by_email(org_id, clerk_user.email)

        if existing_user is not None:
            # Update if name changed
            updated = await self._update_if_changed(existing_user, clerk_user)
            if updated:
                await logger.ainfo(
                    "user_sync_updated",
                    user_id=str(existing_user.id),
                    clerk_id=clerk_user.id,
                    email=clerk_user.email,
                )
            return existing_user

        # Create new user
        new_user = await self._create_user(clerk_user, org_id)
        await logger.ainfo(
            "user_sync_created",
            user_id=str(new_user.id),
            clerk_id=clerk_user.id,
            email=clerk_user.email,
            org_id=str(org_id),
        )
        return new_user

    async def _update_if_changed(
        self,
        existing_user: OrgUser,
        clerk_user: ClerkUser,
    ) -> bool:
        """Update user if Clerk metadata has changed.

        Args:
            existing_user: Existing local user record.
            clerk_user: Current Clerk user info.

        Returns:
            True if user was updated, False otherwise.
        """
        new_name = clerk_user.full_name
        if new_name and existing_user.name != new_name:
            await self._repo.update(
                existing_user.id,
                OrgUserUpdate(name=new_name),
            )
            return True
        return False

    async def _create_user(
        self,
        clerk_user: ClerkUser,
        org_id: UUID,
    ) -> OrgUser:
        """Create a new local user from Clerk user.

        Args:
            clerk_user: Clerk user information.
            org_id: Organization ID.

        Returns:
            Created OrgUser.
        """
        if not clerk_user.email:
            raise ValueError("Clerk user must have an email")

        create_data = OrgUserCreate(
            email=clerk_user.email,
            name=clerk_user.full_name,
            role="member",  # Default role for new users
        )
        return await self._repo.create(org_id, create_data)

    async def get_by_clerk_email(
        self,
        email: str,
        org_id: UUID,
    ) -> OrgUser | None:
        """Find a local user by email.

        Args:
            email: User's email address.
            org_id: Organization ID.

        Returns:
            OrgUser if found, None otherwise.
        """
        return await self._repo.get_by_email(org_id, email)

    async def ensure_user_exists(
        self,
        clerk_user: ClerkUser,
        org_id: UUID,
    ) -> OrgUser:
        """Ensure a local user exists for the Clerk user.

        This is a convenience method that calls sync_user.

        Args:
            clerk_user: User information from Clerk JWT.
            org_id: Organization ID.

        Returns:
            The local OrgUser record (existing or newly created).
        """
        return await self.sync_user(clerk_user, org_id)
