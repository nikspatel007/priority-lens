"""Service for orchestrating Gmail sync operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from rl_emails.integrations.gmail.client import GmailApiError, GmailClient
from rl_emails.integrations.gmail.parser import gmail_to_email_data, parse_raw_message

if TYPE_CHECKING:
    from rl_emails.core.types import EmailData
    from rl_emails.repositories.sync_state import SyncStateRepository


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        success: Whether sync completed successfully.
        emails_synced: Number of emails synced.
        history_id: Gmail history ID for incremental sync.
        error: Error message if sync failed.
    """

    success: bool
    emails_synced: int
    history_id: str | None = None
    error: str | None = None


class SyncService:
    """Orchestrates Gmail sync operations.

    This service handles the high-level sync workflow:
    1. List messages from Gmail
    2. Fetch message content in batches
    3. Parse and convert to EmailData
    4. Track sync state for incremental updates

    Typical usage:
        client = GmailClient(access_token="...")
        sync_repo = SyncStateRepository(session)
        service = SyncService(gmail_client=client, sync_repo=sync_repo)

        # Sync with callback for progress
        def on_progress(processed: int, total: int) -> None:
            print(f"Progress: {processed}/{total}")

        result = await service.initial_sync(
            user_id=uuid,
            days=30,
            progress_callback=on_progress,
        )
    """

    def __init__(
        self,
        gmail_client: GmailClient,
        sync_repo: SyncStateRepository,
    ) -> None:
        """Initialize sync service.

        Args:
            gmail_client: Gmail API client for fetching emails.
            sync_repo: Repository for sync state persistence.
        """
        self.gmail_client = gmail_client
        self.sync_repo = sync_repo

    async def initial_sync(
        self,
        user_id: UUID,
        days: int = 30,
        max_messages: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[SyncResult, list[EmailData]]:
        """Perform initial full sync of Gmail messages.

        Fetches all messages within the date range and converts them
        to EmailData format for pipeline processing.

        Args:
            user_id: User UUID to sync for.
            days: Number of days of history to sync.
            max_messages: Maximum messages to sync (None for all).
            progress_callback: Called with (processed, total) counts.

        Returns:
            Tuple of (SyncResult, list of EmailData).
        """
        # Mark sync as started
        await self.sync_repo.start_sync(user_id)

        try:
            # Build query for date filter
            query = f"newer_than:{days}d"

            # List all message IDs
            message_refs = []
            async for ref in self.gmail_client.list_all_messages(
                query=query,
                max_messages=max_messages,
            ):
                message_refs.append(ref)

            total = len(message_refs)
            if total == 0:
                await self.sync_repo.complete_sync(user_id, None, 0)
                return SyncResult(success=True, emails_synced=0), []

            # Fetch full messages in batches
            emails: list[EmailData] = []
            processed = 0
            latest_history_id: str | None = None

            results = await self.gmail_client.batch_get_messages(message_refs)

            for result in results:
                if isinstance(result, GmailApiError):
                    # Skip failed messages
                    processed += 1
                    continue

                # Parse and convert
                try:
                    gmail_msg = parse_raw_message(result)
                    email_data = gmail_to_email_data(gmail_msg)
                    emails.append(email_data)

                    # Track latest history ID for incremental sync
                    if gmail_msg.history_id:
                        if latest_history_id is None or gmail_msg.history_id > latest_history_id:
                            latest_history_id = gmail_msg.history_id

                except Exception:
                    # Skip unparseable messages
                    pass

                processed += 1
                if progress_callback:
                    progress_callback(processed, total)

            # Complete sync
            await self.sync_repo.complete_sync(user_id, latest_history_id, len(emails))

            return SyncResult(
                success=True,
                emails_synced=len(emails),
                history_id=latest_history_id,
            ), emails

        except GmailApiError as e:
            await self.sync_repo.fail_sync(user_id, str(e))
            return SyncResult(
                success=False,
                emails_synced=0,
                error=str(e),
            ), []

        except Exception as e:
            await self.sync_repo.fail_sync(user_id, str(e))
            return SyncResult(
                success=False,
                emails_synced=0,
                error=str(e),
            ), []

    async def incremental_sync(
        self,
        user_id: UUID,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[SyncResult, list[EmailData]]:
        """Perform incremental sync using Gmail history API.

        Only fetches messages that changed since the last sync.
        Falls back to initial sync if no history ID is available.

        Args:
            user_id: User UUID to sync for.
            progress_callback: Called with (processed, total) counts.

        Returns:
            Tuple of (SyncResult, list of EmailData).
        """
        # Get current sync state
        state = await self.sync_repo.get_by_user_id(user_id)
        if state is None or state.last_history_id is None:
            # No previous sync, do initial sync
            return await self.initial_sync(
                user_id,
                days=30,
                progress_callback=progress_callback,
            )

        # Mark sync as started
        await self.sync_repo.start_sync(user_id)

        try:
            # Get history changes
            all_message_ids: set[str] = set()
            page_token = None
            latest_history_id = state.last_history_id

            while True:
                history_records, next_page, history_id = await self.gmail_client.get_history(
                    start_history_id=state.last_history_id,
                    page_token=page_token,
                )

                # Extract message IDs from history
                for record in history_records:
                    messages_added = record.get("messagesAdded", [])
                    if isinstance(messages_added, list):
                        for msg in messages_added:
                            if isinstance(msg, dict):
                                message_info = msg.get("message", {})
                                if isinstance(message_info, dict):
                                    msg_id = message_info.get("id")
                                    if isinstance(msg_id, str):
                                        all_message_ids.add(msg_id)

                if history_id:
                    latest_history_id = history_id

                if not next_page:
                    break
                page_token = next_page

            if not all_message_ids:
                await self.sync_repo.complete_sync(user_id, latest_history_id, 0)
                return SyncResult(
                    success=True,
                    emails_synced=0,
                    history_id=latest_history_id,
                ), []

            # Create refs for batch fetch
            from rl_emails.integrations.gmail.models import GmailMessageRef

            refs = [GmailMessageRef(id=msg_id, thread_id="") for msg_id in all_message_ids]
            total = len(refs)

            # Fetch full messages
            emails: list[EmailData] = []
            processed = 0

            results = await self.gmail_client.batch_get_messages(refs)

            for result in results:
                if isinstance(result, GmailApiError):
                    processed += 1
                    continue

                try:
                    gmail_msg = parse_raw_message(result)
                    email_data = gmail_to_email_data(gmail_msg)
                    emails.append(email_data)

                    if gmail_msg.history_id and gmail_msg.history_id > latest_history_id:
                        latest_history_id = gmail_msg.history_id

                except Exception:
                    pass

                processed += 1
                if progress_callback:
                    progress_callback(processed, total)

            await self.sync_repo.complete_sync(user_id, latest_history_id, len(emails))

            return SyncResult(
                success=True,
                emails_synced=len(emails),
                history_id=latest_history_id,
            ), emails

        except GmailApiError as e:
            # If history is too old, fall back to initial sync
            if e.status_code == 404:
                await self.sync_repo.fail_sync(user_id, "History expired")
                return await self.initial_sync(
                    user_id,
                    days=30,
                    progress_callback=progress_callback,
                )

            await self.sync_repo.fail_sync(user_id, str(e))
            return SyncResult(
                success=False,
                emails_synced=0,
                error=str(e),
            ), []

        except Exception as e:
            await self.sync_repo.fail_sync(user_id, str(e))
            return SyncResult(
                success=False,
                emails_synced=0,
                error=str(e),
            ), []

    async def get_sync_status(self, user_id: UUID) -> dict[str, object]:
        """Get current sync status for a user.

        Args:
            user_id: User UUID.

        Returns:
            Dict with sync status information.
        """
        state = await self.sync_repo.get_by_user_id(user_id)
        if state is None:
            return {
                "status": "not_synced",
                "emails_synced": 0,
                "last_sync_at": None,
                "error": None,
            }

        return {
            "status": state.sync_status,
            "emails_synced": state.emails_synced,
            "last_sync_at": state.last_sync_at.isoformat() if state.last_sync_at else None,
            "last_history_id": state.last_history_id,
            "error": state.error_message,
        }
