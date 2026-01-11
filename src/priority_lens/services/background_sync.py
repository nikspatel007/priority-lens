"""Background sync runner for first-time user email sync.

This service triggers background email sync when a user connects their
Google account for the first time. Uses Celery for reliable task execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from priority_lens.repositories.sync_state import SyncStateRepository

logger = structlog.get_logger(__name__)

# Track active task IDs per user (in-memory cache, cleared on restart)
_user_task_ids: dict[UUID, str] = {}


SyncStatusValue = Literal["pending", "syncing", "completed", "failed"]


@dataclass
class BackgroundSyncStatus:
    """Status of a background sync operation."""

    status: SyncStatusValue
    emails_synced: int
    total_emails: int | None
    progress: float  # 0.0 to 1.0
    error: str | None = None
    task_id: str | None = None


class BackgroundSyncRunner:
    """Runs email sync in background for first-time users.

    This service:
    1. Checks if user needs initial sync
    2. Dispatches Celery task for background processing
    3. Provides status for mobile app polling

    Example:
        runner = BackgroundSyncRunner(sync_repo)

        # Check if sync needed and start if so
        sync_started = await runner.start_if_needed(user_id)

        # Poll for status
        status = await runner.get_status(user_id)
    """

    def __init__(self, sync_repo: SyncStateRepository) -> None:
        """Initialize background sync runner.

        Args:
            sync_repo: Repository for sync state.
        """
        self.sync_repo = sync_repo

    async def needs_initial_sync(self, user_id: UUID) -> bool:
        """Check if user needs initial sync.

        Returns True if:
        - No sync_state record exists
        - sync_status is still "idle" with no emails synced

        Args:
            user_id: User to check.

        Returns:
            True if initial sync is needed.
        """
        state = await self.sync_repo.get_by_user_id(user_id)
        if state is None:
            return True

        # If never synced (no history ID and no emails)
        if state.last_history_id is None and state.emails_synced == 0:
            return True

        return False

    async def is_sync_running(self, user_id: UUID) -> bool:
        """Check if a sync is currently running for user.

        Args:
            user_id: User to check.

        Returns:
            True if sync is running.
        """
        # Check Celery task status if we have a task ID
        if user_id in _user_task_ids:
            from priority_lens.tasks.sync_tasks import get_sync_task_status

            status = get_sync_task_status(_user_task_ids[user_id])
            if status.status == "syncing":
                return True

        # Fall back to database state
        state = await self.sync_repo.get_by_user_id(user_id)
        return state is not None and state.sync_status == "syncing"

    async def start_sync(
        self,
        user_id: UUID,
        max_messages: int = 1000,
        days: int = 15,
    ) -> bool:
        """Start background sync for a user.

        Dispatches a Celery task to handle the sync asynchronously.

        Args:
            user_id: User to sync for.
            max_messages: Maximum emails to sync.
            days: Number of days to look back.

        Returns:
            True if sync was started, False if already running.
        """
        # Check if already running
        if await self.is_sync_running(user_id):
            await logger.ainfo(
                "sync_already_running",
                user_id=str(user_id),
            )
            return False

        # Import here to avoid circular imports
        from priority_lens.tasks.sync_tasks import run_initial_sync

        # Dispatch Celery task
        result = run_initial_sync.delay(
            user_id=str(user_id),
            max_messages=max_messages,
            days=days,
        )

        # Track task ID
        _user_task_ids[user_id] = result.id

        await logger.ainfo(
            "background_sync_dispatched",
            user_id=str(user_id),
            task_id=result.id,
            max_messages=max_messages,
            days=days,
        )

        return True

    async def start_if_needed(
        self,
        user_id: UUID,
        max_messages: int = 1000,
        days: int = 15,
    ) -> bool:
        """Start sync only if needed for first-time user.

        Args:
            user_id: User to check and sync.
            max_messages: Maximum emails to sync.
            days: Number of days to look back.

        Returns:
            True if sync was started, False if not needed or already running.
        """
        needs_sync = await self.needs_initial_sync(user_id)
        if not needs_sync:
            await logger.ainfo(
                "sync_not_needed",
                user_id=str(user_id),
            )
            return False

        return await self.start_sync(
            user_id=user_id,
            max_messages=max_messages,
            days=days,
        )

    async def get_status(self, user_id: UUID) -> BackgroundSyncStatus:
        """Get current sync status for a user.

        Combines Celery task status with database state.

        Args:
            user_id: User to check.

        Returns:
            BackgroundSyncStatus with current state.
        """
        # Check Celery task status if we have a task ID
        if user_id in _user_task_ids:
            from priority_lens.tasks.sync_tasks import get_sync_task_status

            task_status = get_sync_task_status(_user_task_ids[user_id])

            # If task is complete or failed, we can clear the cached ID
            if task_status.status in ("completed", "failed"):
                del _user_task_ids[user_id]

            return BackgroundSyncStatus(
                status=task_status.status,
                emails_synced=task_status.emails_synced,
                total_emails=task_status.total_emails,
                progress=task_status.progress,
                error=task_status.error,
                task_id=task_status.task_id,
            )

        # Fall back to database state
        state = await self.sync_repo.get_by_user_id(user_id)

        if state is None:
            return BackgroundSyncStatus(
                status="pending",
                emails_synced=0,
                total_emails=None,
                progress=0.0,
            )

        # Map internal status to API status
        status: SyncStatusValue
        if state.sync_status == "syncing":
            status = "syncing"
        elif state.sync_status == "error":
            status = "failed"
        elif state.last_sync_at is not None:
            status = "completed"
        else:
            status = "pending"

        # Calculate progress (rough estimate)
        progress = 0.0
        if status == "completed":
            progress = 1.0
        elif status == "syncing" and state.emails_synced > 0:
            # Estimate progress based on typical email count
            progress = min(state.emails_synced / 1000.0, 0.99)

        return BackgroundSyncStatus(
            status=status,
            emails_synced=state.emails_synced,
            total_emails=None,
            progress=progress,
            error=state.error_message if status == "failed" else None,
        )
