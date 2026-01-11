"""Background sync tasks for email processing.

This module defines Celery tasks for:
- Initial email sync (first-time users)
- Progressive email processing pipeline
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import structlog
from celery import Task  # type: ignore[import-untyped]
from celery.result import AsyncResult  # type: ignore[import-untyped]

from priority_lens.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


SyncStatusValue = Literal["pending", "syncing", "completed", "failed"]


@dataclass
class SyncTaskStatus:
    """Status of a sync task."""

    status: SyncStatusValue
    emails_synced: int
    total_emails: int | None
    progress: float  # 0.0 to 1.0
    error: str | None = None
    task_id: str | None = None


class SyncTask(Task):  # type: ignore[misc]
    """Base class for sync tasks with error handling."""

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 60}
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes max


def _run_async(coro: Any) -> Any:
    """Run an async coroutine in a sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, base=SyncTask, name="sync.initial")  # type: ignore[untyped-decorator]
def run_initial_sync(
    self: Task,
    user_id: str,
    max_messages: int = 1000,
    days: int = 15,
) -> dict[str, Any]:
    """Run initial email sync for a user.

    This task:
    1. Fetches emails from Gmail API
    2. Stores them in the database
    3. Runs the analysis pipeline (embeddings, classification, etc.)

    Args:
        user_id: UUID string of the user.
        max_messages: Maximum emails to sync.
        days: Number of days to look back.

    Returns:
        Dict with sync results.
    """
    result: dict[str, Any] = _run_async(_run_initial_sync_async(self, user_id, max_messages, days))
    return result


async def _run_initial_sync_async(
    task: Task,
    user_id_str: str,
    max_messages: int,
    days: int,
) -> dict[str, Any]:
    """Async implementation of initial sync."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from priority_lens.core.config import Config
    from priority_lens.integrations.gmail.client import GmailClient
    from priority_lens.repositories.oauth_token import OAuthTokenRepository
    from priority_lens.repositories.sync_state import SyncStateRepository
    from priority_lens.services.progressive_sync import (
        PhaseConfig,
        ProgressiveSyncService,
        SyncPhase,
    )

    user_id = UUID(user_id_str)
    config = Config.from_env()  # Loads from environment

    await logger.ainfo(
        "sync_task_started",
        user_id=user_id_str,
        task_id=task.request.id,
        max_messages=max_messages,
        days=days,
    )

    # Create database connection (use async URL for SQLAlchemy async)
    engine = create_async_engine(config.async_database_url)
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_factory() as session:
        try:
            # Get OAuth tokens for user
            token_repo = OAuthTokenRepository(session)
            sync_repo = SyncStateRepository(session)

            token = await token_repo.get_by_user(user_id, "google")
            if not token:
                await logger.aerror(
                    "sync_task_no_token",
                    user_id=user_id_str,
                )
                return {
                    "status": "failed",
                    "error": "No valid OAuth token found",
                    "emails_synced": 0,
                }

            # Mark sync as started
            await sync_repo.start_sync(user_id)
            await session.commit()

            # Update task state for progress tracking
            task.update_state(
                state="PROGRESS",
                meta={"status": "syncing", "emails_synced": 0, "progress": 0.0},
            )

            # Create Gmail client
            async with GmailClient(access_token=token.access_token) as gmail_client:
                # Create progressive sync service
                progressive_service = ProgressiveSyncService(
                    gmail_client=gmail_client,
                    sync_repo=sync_repo,
                    session=session,
                )

                # Configure sync phase
                quick_phase = PhaseConfig(
                    phase=SyncPhase.QUICK,
                    days_start=0,
                    days_end=days,
                    batch_size=100,
                    run_embeddings=True,
                    run_llm=True,
                    llm_limit=100,
                )

                total_processed = 0
                async for progress in progressive_service.sync_progressive(
                    user_id=user_id,
                    config=config,
                    phases=[quick_phase],
                    max_messages=max_messages,
                ):
                    total_processed = progress.emails_processed

                    # Update task progress
                    task.update_state(
                        state="PROGRESS",
                        meta={
                            "status": "syncing",
                            "emails_synced": total_processed,
                            "progress": min(total_processed / max_messages, 0.99),
                        },
                    )

                    if progress.error:
                        await sync_repo.fail_sync(user_id, progress.error)
                        await session.commit()
                        await logger.aerror(
                            "sync_task_failed",
                            user_id=user_id_str,
                            error=progress.error,
                        )
                        return {
                            "status": "failed",
                            "error": progress.error,
                            "emails_synced": total_processed,
                        }

            # Mark sync as complete
            await sync_repo.complete_sync(
                user_id=user_id,
                history_id=None,
                emails_synced=total_processed,
            )
            await session.commit()

            await logger.ainfo(
                "sync_task_completed",
                user_id=user_id_str,
                emails_synced=total_processed,
            )

            return {
                "status": "completed",
                "emails_synced": total_processed,
                "error": None,
            }

        except Exception as e:
            await logger.aerror(
                "sync_task_error",
                user_id=user_id_str,
                error=str(e),
            )
            # Mark sync as failed
            await sync_repo.fail_sync(user_id, str(e))
            await session.commit()
            raise


def get_sync_task_status(task_id: str) -> SyncTaskStatus:
    """Get the status of a sync task.

    Args:
        task_id: Celery task ID.

    Returns:
        SyncTaskStatus with current state.
    """
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return SyncTaskStatus(
            status="pending",
            emails_synced=0,
            total_emails=None,
            progress=0.0,
            task_id=task_id,
        )
    elif result.state == "PROGRESS":
        info = result.info or {}
        return SyncTaskStatus(
            status="syncing",
            emails_synced=info.get("emails_synced", 0),
            total_emails=None,
            progress=info.get("progress", 0.0),
            task_id=task_id,
        )
    elif result.state == "SUCCESS":
        info = result.result or {}
        return SyncTaskStatus(
            status="completed",
            emails_synced=info.get("emails_synced", 0),
            total_emails=None,
            progress=1.0,
            task_id=task_id,
        )
    elif result.state == "FAILURE":
        return SyncTaskStatus(
            status="failed",
            emails_synced=0,
            total_emails=None,
            progress=0.0,
            error=str(result.result) if result.result else "Unknown error",
            task_id=task_id,
        )
    else:
        # STARTED or other states
        return SyncTaskStatus(
            status="syncing",
            emails_synced=0,
            total_emails=None,
            progress=0.0,
            task_id=task_id,
        )
