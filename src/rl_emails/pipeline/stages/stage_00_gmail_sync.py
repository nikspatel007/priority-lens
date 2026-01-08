"""Stage 00: Gmail API sync for multi-tenant mode.

This stage syncs emails from Gmail API and stores them in the database.
It replaces stages 1-2 (parse_mbox, import_postgres) for Gmail-based ingestion.

Only runs in multi-tenant mode when a user_id is configured.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from rl_emails.pipeline.stages.base import StageResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rl_emails.core.config import Config
    from rl_emails.core.types import EmailData


def run(config: Config, days: int = 30, max_messages: int | None = None) -> StageResult:
    """Run Gmail sync stage.

    This stage syncs emails from Gmail API for the configured user.
    Only available in multi-tenant mode.

    Args:
        config: Pipeline configuration with user_id set.
        days: Number of days of email history to sync.
        max_messages: Maximum messages to sync (None for all).

    Returns:
        StageResult indicating success/failure and email count.
    """
    start_time = time.time()

    if not config.is_multi_tenant:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=time.time() - start_time,
            message="Gmail sync requires multi-tenant mode (--user flag)",
        )

    if config.user_id is None:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=time.time() - start_time,
            message="Gmail sync requires user_id in config",
        )

    # Run async sync
    try:
        result = asyncio.run(_run_async(config, days, max_messages))
        return StageResult(
            success=result.success,
            records_processed=result.records_processed,
            duration_seconds=time.time() - start_time,
            message=result.message,
            metadata=result.metadata,
        )
    except Exception as e:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=time.time() - start_time,
            message=str(e),
        )


async def _run_async(config: Config, days: int, max_messages: int | None) -> StageResult:
    """Async implementation of Gmail sync.

    Args:
        config: Pipeline configuration.
        days: Number of days to sync.
        max_messages: Maximum messages to sync.

    Returns:
        StageResult with sync outcome.
    """
    # Import here to avoid circular imports
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from rl_emails.integrations.gmail.client import GmailClient
    from rl_emails.repositories.oauth_token import OAuthTokenRepository
    from rl_emails.repositories.sync_state import SyncStateRepository
    from rl_emails.services.sync_service import SyncService

    start_time = time.time()

    if config.user_id is None:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=time.time() - start_time,
            message="user_id is required",
        )

    # Create async database connection - integration code  # pragma: no cover
    db_url = config.database_url  # pragma: no cover
    if db_url.startswith("postgresql://"):  # pragma: no cover
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)  # pragma: no cover

    engine = create_async_engine(db_url)  # pragma: no cover

    async with AsyncSession(engine) as session:  # pragma: no cover
        # Get OAuth token
        token_repo = OAuthTokenRepository(session)  # pragma: no cover
        token = await token_repo.get_by_user(config.user_id)  # pragma: no cover

        if token is None:  # pragma: no cover
            return StageResult(  # pragma: no cover
                success=False,
                records_processed=0,
                duration_seconds=time.time() - start_time,
                message="No valid OAuth token found. Run 'rl-emails auth connect' first.",
            )

        # Create Gmail client and sync service
        async with GmailClient(access_token=token.access_token) as gmail_client:  # pragma: no cover
            sync_repo = SyncStateRepository(session)  # pragma: no cover
            sync_service = SyncService(  # pragma: no cover
                gmail_client=gmail_client,
                sync_repo=sync_repo,
            )

            # Perform sync
            sync_result, emails = await sync_service.initial_sync(  # pragma: no cover
                user_id=config.user_id,
                days=days,
                max_messages=max_messages,
            )

            if not sync_result.success:  # pragma: no cover
                return StageResult(  # pragma: no cover
                    success=False,
                    records_processed=0,
                    duration_seconds=time.time() - start_time,
                    message=sync_result.error or "Sync failed",
                )

            # Store emails in database
            if emails:  # pragma: no cover
                stored_count = await _store_emails(session, config, emails)  # pragma: no cover
            else:  # pragma: no cover
                stored_count = 0  # pragma: no cover

            return StageResult(  # pragma: no cover
                success=True,
                records_processed=stored_count,
                duration_seconds=time.time() - start_time,
                message=f"Synced {sync_result.emails_synced} emails, stored {stored_count}",
                metadata={
                    "history_id": sync_result.history_id,
                    "synced": sync_result.emails_synced,
                    "stored": stored_count,
                },
            )


async def _store_emails(
    session: AsyncSession,
    config: Config,
    emails: list[EmailData],
) -> int:
    """Store synced emails in the database.

    This inserts emails into the raw_emails and emails tables,
    similar to stage_02_import_postgres but for Gmail API data.

    Args:
        session: Database session.
        config: Pipeline configuration.
        emails: List of EmailData dicts to store.

    Returns:
        Number of emails stored.
    """
    # Avoid unused parameter warning
    _ = config

    from datetime import datetime

    from sqlalchemy import text

    stored = 0

    for email_data in emails:
        try:
            # Extract fields with defaults
            message_id = str(email_data.get("message_id", ""))
            if not message_id:
                continue

            # Gmail-specific IDs for API operations and threading
            gmail_id = email_data.get("gmail_id")
            thread_id = email_data.get("thread_id")

            subject = str(email_data.get("subject", ""))
            from_email = str(email_data.get("from_email", ""))
            from_name = email_data.get("from_name")
            date_str = str(email_data.get("date_str", ""))
            body_text = str(email_data.get("body_text", ""))
            body_html = email_data.get("body_html")
            labels_raw = email_data.get("labels", [])
            to_emails_raw = email_data.get("to_emails", [])
            cc_emails_raw = email_data.get("cc_emails", [])
            in_reply_to = email_data.get("in_reply_to")
            references_raw = email_data.get("references", [])

            # Safely cast to lists
            labels: list[str] = list(labels_raw) if isinstance(labels_raw, list) else []
            to_emails: list[str] = list(to_emails_raw) if isinstance(to_emails_raw, list) else []
            cc_emails: list[str] = list(cc_emails_raw) if isinstance(cc_emails_raw, list) else []
            references: list[str] = list(references_raw) if isinstance(references_raw, list) else []

            # Parse date
            date_parsed = None
            if date_str:
                try:
                    date_parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Insert into raw_emails
            raw_result = await session.execute(
                text(
                    """
                    INSERT INTO raw_emails (
                        message_id, in_reply_to, references_raw,
                        date_raw, from_raw, to_raw, cc_raw,
                        subject_raw, body_text, body_html, labels_raw
                    ) VALUES (
                        :message_id, :in_reply_to, :references_raw,
                        :date_raw, :from_raw, :to_raw, :cc_raw,
                        :subject_raw, :body_text, :body_html, :labels_raw
                    )
                    ON CONFLICT (message_id) DO NOTHING
                    RETURNING id
                """
                ),
                {
                    "message_id": message_id,
                    "in_reply_to": in_reply_to,
                    "references_raw": " ".join(references) if references else None,
                    "date_raw": date_str or None,
                    "from_raw": f"{from_name} <{from_email}>" if from_name else from_email,
                    "to_raw": ", ".join(to_emails) if to_emails else None,
                    "cc_raw": ", ".join(cc_emails) if cc_emails else None,
                    "subject_raw": subject,
                    "body_text": body_text,
                    "body_html": body_html,
                    "labels_raw": ",".join(labels) if labels else None,
                },
            )

            raw_row = raw_result.fetchone()
            if raw_row is None:
                # Already exists
                continue

            raw_id = raw_row[0]

            # Check if sent
            is_sent = "SENT" in labels

            # Insert into emails
            await session.execute(
                text(
                    """
                    INSERT INTO emails (
                        raw_email_id, message_id, gmail_id, thread_id,
                        in_reply_to, date_parsed,
                        from_email, from_name, to_emails, cc_emails, subject,
                        body_text, body_preview, word_count, labels,
                        has_attachments, is_sent, enriched_at
                    ) VALUES (
                        :raw_email_id, :message_id, :gmail_id, :thread_id,
                        :in_reply_to, :date_parsed,
                        :from_email, :from_name, :to_emails, :cc_emails, :subject,
                        :body_text, :body_preview, :word_count, :labels,
                        :has_attachments, :is_sent, NOW()
                    )
                    ON CONFLICT (message_id) DO NOTHING
                """
                ),
                {
                    "raw_email_id": raw_id,
                    "message_id": message_id,
                    "gmail_id": gmail_id,
                    "thread_id": thread_id,
                    "in_reply_to": in_reply_to,
                    "date_parsed": date_parsed,
                    "from_email": from_email,
                    "from_name": from_name,
                    "to_emails": to_emails if to_emails else None,
                    "cc_emails": cc_emails if cc_emails else None,
                    "subject": subject,
                    "body_text": body_text,
                    "body_preview": body_text[:200] if body_text else None,
                    "word_count": len(body_text.split()) if body_text else 0,
                    "labels": labels if labels else None,
                    "has_attachments": False,  # Could be enhanced
                    "is_sent": is_sent,
                },
            )

            stored += 1

        except Exception:
            # Skip problematic emails
            continue

    await session.commit()
    return stored
