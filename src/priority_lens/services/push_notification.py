"""Service for handling Gmail push notifications."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from priority_lens.repositories.org_user import OrgUserRepository
    from priority_lens.repositories.sync_state import SyncStateRepository
    from priority_lens.repositories.watch_subscription import WatchSubscriptionRepository
    from priority_lens.services.sync_service import SyncService

logger = structlog.get_logger(__name__)


class PushNotificationError(Exception):
    """Base exception for push notification errors."""

    pass


class InvalidNotificationError(PushNotificationError):
    """Raised when notification data is invalid."""

    pass


class UserNotFoundError(PushNotificationError):
    """Raised when user cannot be found for notification."""

    pass


@dataclass
class NotificationData:
    """Parsed push notification data.

    Attributes:
        email_address: Gmail address that received the update.
        history_id: Gmail history ID for incremental sync.
        raw_data: Original decoded notification payload.
    """

    email_address: str
    history_id: str
    raw_data: dict[str, object] = field(default_factory=dict)


@dataclass
class NotificationResult:
    """Result of processing a notification.

    Attributes:
        success: Whether processing completed successfully.
        user_id: User ID that was synced (if found).
        emails_synced: Number of new emails synced.
        skipped: Whether notification was skipped (e.g., duplicate).
        error: Error message if processing failed.
    """

    success: bool
    user_id: UUID | None = None
    emails_synced: int = 0
    skipped: bool = False
    error: str | None = None


class NotificationDeduplicator:
    """Simple in-memory deduplicator for push notifications.

    Gmail may send duplicate notifications for the same change.
    This class tracks recently processed history IDs to avoid
    redundant sync operations.
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000) -> None:
        """Initialize deduplicator.

        Args:
            ttl_seconds: How long to remember processed notifications.
            max_entries: Maximum entries to track (LRU eviction).
        """
        self._cache: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._max_entries = max_entries

    def is_duplicate(self, key: str) -> bool:
        """Check if this notification was recently processed.

        Args:
            key: Unique key for the notification (email + history_id).

        Returns:
            True if this is a duplicate, False otherwise.
        """
        self._cleanup_expired()

        if key in self._cache:
            return True

        return False

    def mark_processed(self, key: str) -> None:
        """Mark a notification as processed.

        Args:
            key: Unique key for the notification.
        """
        self._cleanup_expired()

        # Evict oldest entries if at capacity
        while len(self._cache) >= self._max_entries:
            oldest = min(self._cache, key=lambda k: self._cache[k])
            del self._cache[oldest]

        self._cache[key] = time.time()

    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v > self._ttl]
        for key in expired:
            del self._cache[key]


class PushNotificationService:
    """Handles Gmail push notifications via Pub/Sub.

    This service processes notifications from Google Cloud Pub/Sub,
    validates them, and triggers incremental syncs for affected users.

    Typical usage:
        service = PushNotificationService(
            user_repo=user_repo,
            sync_repo=sync_repo,
            watch_repo=watch_repo,
            sync_service=sync_service,
        )

        # Handle webhook request
        result = await service.handle_notification(pubsub_message)
    """

    def __init__(
        self,
        user_repo: OrgUserRepository,
        sync_repo: SyncStateRepository,
        watch_repo: WatchSubscriptionRepository,
        sync_service: SyncService | None = None,
    ) -> None:
        """Initialize push notification service.

        Args:
            user_repo: Repository for user lookup.
            sync_repo: Repository for sync state.
            watch_repo: Repository for watch subscriptions.
            sync_service: Service for triggering sync (optional, can be set later).
        """
        self.user_repo = user_repo
        self.sync_repo = sync_repo
        self.watch_repo = watch_repo
        self.sync_service = sync_service
        self._deduplicator = NotificationDeduplicator()

    def set_sync_service(self, sync_service: SyncService) -> None:
        """Set the sync service for triggering syncs.

        Args:
            sync_service: Service for running sync operations.
        """
        self.sync_service = sync_service

    def parse_notification(self, message_data: str) -> NotificationData:
        """Parse a Pub/Sub message into notification data.

        The message data is base64-encoded JSON from Gmail.

        Args:
            message_data: Base64-encoded Pub/Sub message data.

        Returns:
            Parsed notification data.

        Raises:
            InvalidNotificationError: If message cannot be parsed.
        """
        try:
            # Decode base64
            decoded = base64.b64decode(message_data)
            payload = json.loads(decoded)
        except (ValueError, json.JSONDecodeError) as e:
            raise InvalidNotificationError(f"Failed to decode notification: {e}") from e

        # Extract required fields
        email_address = payload.get("emailAddress")
        history_id = payload.get("historyId")

        if not email_address:
            raise InvalidNotificationError("Missing emailAddress in notification")
        if not history_id:
            raise InvalidNotificationError("Missing historyId in notification")

        return NotificationData(
            email_address=str(email_address),
            history_id=str(history_id),
            raw_data=payload,
        )

    async def handle_notification(self, message_data: str) -> NotificationResult:
        """Handle a Gmail push notification.

        This method:
        1. Parses the notification data
        2. Finds the user by email
        3. Checks for duplicates
        4. Triggers incremental sync

        Args:
            message_data: Base64-encoded Pub/Sub message data.

        Returns:
            NotificationResult with processing outcome.
        """
        # Parse notification
        try:
            notification = self.parse_notification(message_data)
        except InvalidNotificationError as e:
            await logger.aerror("invalid_notification", error=str(e))
            return NotificationResult(success=False, error=str(e))

        await logger.ainfo(
            "notification_received",
            email=notification.email_address,
            history_id=notification.history_id,
        )

        # Find user by email across all organizations
        user = await self.user_repo.find_by_email(notification.email_address)
        if user is None:
            await logger.awarning(
                "user_not_found_for_notification",
                email=notification.email_address,
            )
            return NotificationResult(
                success=False,
                error=f"User not found: {notification.email_address}",
            )

        # Check for duplicates
        dedup_key = f"{notification.email_address}:{notification.history_id}"
        if self._deduplicator.is_duplicate(dedup_key):
            await logger.ainfo(
                "duplicate_notification_skipped",
                email=notification.email_address,
                history_id=notification.history_id,
            )
            return NotificationResult(success=True, user_id=user.id, skipped=True)

        # Record notification
        await self.watch_repo.record_notification(user.id)

        # Mark as processed before sync to avoid re-processing on failure
        self._deduplicator.mark_processed(dedup_key)

        # Trigger incremental sync if sync service available
        emails_synced = 0
        if self.sync_service:
            try:
                result, _emails = await self.sync_service.incremental_sync(user.id)
                emails_synced = result.emails_synced
            except Exception as e:
                await logger.aerror(
                    "sync_failed_after_notification",
                    user_id=str(user.id),
                    error=str(e),
                )
                return NotificationResult(
                    success=False,
                    user_id=user.id,
                    error=f"Sync failed: {e}",
                )

        await logger.ainfo(
            "notification_processed",
            user_id=str(user.id),
            emails_synced=emails_synced,
        )

        return NotificationResult(
            success=True,
            user_id=user.id,
            emails_synced=emails_synced,
        )

    async def handle_notification_batch(self, messages: list[str]) -> list[NotificationResult]:
        """Handle multiple notifications.

        Args:
            messages: List of base64-encoded Pub/Sub message data.

        Returns:
            List of NotificationResult for each message.
        """
        results = []
        for message_data in messages:
            result = await self.handle_notification(message_data)
            results.append(result)
        return results
