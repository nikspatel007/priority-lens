"""Gmail email provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from priority_lens.integrations.gmail.client import GmailApiError, GmailClient
from priority_lens.integrations.gmail.models import GmailMessageRef
from priority_lens.integrations.gmail.parser import gmail_to_email_data, parse_raw_message
from priority_lens.providers.base import (
    AuthorizationError,
    ConnectionError,
    ConnectionState,
    ConnectionStatus,
    EmailProvider,
    ProviderType,
    SyncError,
    SyncProgress,
)

if TYPE_CHECKING:
    from priority_lens.core.types import EmailData
    from priority_lens.repositories.oauth_token import OAuthTokenRepository
    from priority_lens.repositories.sync_state import SyncStateRepository
    from priority_lens.repositories.watch_subscription import WatchSubscriptionRepository
    from priority_lens.services.auth_service import AuthService

logger = structlog.get_logger(__name__)


class GmailProvider(EmailProvider):
    """Gmail email provider implementation.

    Wraps the existing Gmail integration to provide a unified interface.
    Uses AuthService for OAuth and SyncStateRepository for sync tracking.

    Example:
        provider = GmailProvider(auth_service, token_repo, sync_repo)
        status = await provider.get_status(user_id)

        if not status.is_connected:
            auth_url = await provider.get_auth_url()
            # redirect user to auth_url
    """

    def __init__(
        self,
        auth_service: AuthService,
        token_repo: OAuthTokenRepository,
        sync_repo: SyncStateRepository,
    ) -> None:
        """Initialize Gmail provider.

        Args:
            auth_service: Service for OAuth flow orchestration.
            token_repo: Repository for token storage.
            sync_repo: Repository for sync state tracking.
        """
        self._auth_service = auth_service
        self._token_repo = token_repo
        self._sync_repo = sync_repo
        self._active_syncs: dict[UUID, SyncProgress] = {}

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type identifier."""
        return ProviderType.GMAIL

    async def get_auth_url(self, state: str | None = None) -> str:
        """Get OAuth authorization URL for Gmail.

        Args:
            state: Optional state parameter for CSRF protection.

        Returns:
            URL to redirect user for Gmail authorization.
        """
        return self._auth_service.start_auth_flow(state=state)

    async def complete_auth(
        self, user_id: UUID, code: str, *, from_mobile: bool = False
    ) -> ConnectionStatus:
        """Complete Gmail OAuth flow.

        Args:
            user_id: User to connect for.
            code: Authorization code from OAuth callback.
            from_mobile: If True, this is a serverAuthCode from mobile OAuth.

        Returns:
            Connection status after authorization.

        Raises:
            AuthorizationError: If authorization fails.
        """
        try:
            token = await self._auth_service.complete_auth_flow(
                user_id, code, from_mobile=from_mobile
            )

            await logger.ainfo(
                "gmail_connected",
                user_id=str(user_id),
                expires_at=str(token.expires_at),
            )

            return ConnectionStatus(
                provider=ProviderType.GMAIL,
                state=ConnectionState.CONNECTED,
                connected_at=datetime.now(UTC),
                metadata={"scopes": token.scopes},
            )

        except Exception as e:
            await logger.aerror(
                "gmail_auth_failed",
                user_id=str(user_id),
                error=str(e),
            )
            raise AuthorizationError(
                f"Gmail authorization failed: {e}",
                provider=ProviderType.GMAIL,
            ) from e

    async def disconnect(self, user_id: UUID) -> bool:
        """Disconnect user from Gmail.

        Args:
            user_id: User to disconnect.

        Returns:
            True if disconnected, False if wasn't connected.
        """
        result = await self._auth_service.revoke_token(user_id, provider="google")

        if result:
            await logger.ainfo("gmail_disconnected", user_id=str(user_id))

        return result

    async def get_status(self, user_id: UUID) -> ConnectionStatus:
        """Get Gmail connection status for user.

        Args:
            user_id: User to check status for.

        Returns:
            Current connection status.
        """
        token = await self._token_repo.get_by_user(user_id, provider="google")

        if token is None:
            return ConnectionStatus(
                provider=ProviderType.GMAIL,
                state=ConnectionState.DISCONNECTED,
            )

        # Check if token is expired
        if token.is_expired:
            return ConnectionStatus(
                provider=ProviderType.GMAIL,
                state=ConnectionState.ERROR,
                error="Token expired, reconnection needed",
                connected_at=token.created_at,
                metadata={"scopes": token.scopes},
            )

        # Get last sync time if available
        sync_state = await self._sync_repo.get_by_user_id(user_id)
        last_sync = sync_state.last_sync_at if sync_state else None

        return ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
            connected_at=token.created_at,
            last_sync=last_sync,
            metadata={"scopes": token.scopes},
        )

    async def sync_messages(
        self,
        user_id: UUID,
        days: int | None = None,
        max_messages: int | None = None,
    ) -> AsyncIterator[EmailData]:
        """Sync messages from Gmail.

        Args:
            user_id: User to sync for.
            days: Number of days to sync (default 30).
            max_messages: Maximum messages to sync (None for all).

        Yields:
            EmailData for each synced message.

        Raises:
            SyncError: If sync fails.
            ConnectionError: If not connected.
        """
        # Get valid token
        try:
            access_token = await self._auth_service.get_valid_token(user_id, "google")
        except Exception as e:
            raise ConnectionError(
                f"Not connected to Gmail: {e}",
                provider=ProviderType.GMAIL,
            ) from e

        # Build query
        sync_days = days or 30
        query = f"newer_than:{sync_days}d"

        # Mark sync started
        await self._sync_repo.start_sync(user_id)

        try:
            async with GmailClient(access_token) as client:
                # List messages - collect all matching messages
                message_refs: list[GmailMessageRef] = []
                try:
                    async for ref in client.list_all_messages(
                        query=query,
                        max_messages=max_messages,
                    ):
                        message_refs.append(ref)
                except GmailApiError as e:
                    raise SyncError(
                        f"Failed to list messages: {e}",
                        provider=ProviderType.GMAIL,
                    ) from e

                total = len(message_refs)
                self._active_syncs[user_id] = SyncProgress(
                    processed=0,
                    total=total,
                    current_phase="fetching",
                )

                await logger.ainfo(
                    "gmail_sync_started",
                    user_id=str(user_id),
                    total_messages=total,
                    days=sync_days,
                )

                # Fetch and yield messages
                for i, ref in enumerate(message_refs):
                    try:
                        raw_message = await client.get_message(ref.id)
                        gmail_message = parse_raw_message(raw_message)
                        email_data = gmail_to_email_data(gmail_message)

                        self._active_syncs[user_id] = SyncProgress(
                            processed=i + 1,
                            total=total,
                            current_phase="processing",
                        )

                        yield email_data

                    except Exception as e:
                        await logger.awarning(
                            "gmail_message_parse_error",
                            message_id=ref.id,
                            error=str(e),
                        )
                        # Continue with next message

                # Get history ID for incremental sync
                profile = await client.get_profile()
                history_id = profile.get("historyId")
                history_id_str = str(history_id) if history_id else None

                # Mark sync complete
                await self._sync_repo.complete_sync(
                    user_id=user_id,
                    history_id=history_id_str,
                    emails_synced=total,
                )

                await logger.ainfo(
                    "gmail_sync_completed",
                    user_id=str(user_id),
                    messages_synced=total,
                    history_id=history_id,
                )

        except (ConnectionError, SyncError):
            raise
        except Exception as e:
            await self._sync_repo.fail_sync(user_id, str(e))
            raise SyncError(
                f"Sync failed: {e}",
                provider=ProviderType.GMAIL,
            ) from e
        finally:
            # Clean up progress tracking
            self._active_syncs.pop(user_id, None)

    async def get_sync_progress(self, user_id: UUID) -> SyncProgress | None:
        """Get current sync progress if sync is in progress.

        Args:
            user_id: User to check progress for.

        Returns:
            Sync progress if sync is active, None otherwise.
        """
        return self._active_syncs.get(user_id)

    def set_watch_repo(self, watch_repo: WatchSubscriptionRepository) -> None:
        """Set the watch subscription repository.

        This allows adding watch functionality after initialization.

        Args:
            watch_repo: Repository for watch subscription tracking.
        """
        self._watch_repo = watch_repo

    async def setup_watch(
        self,
        user_id: UUID,
        topic_name: str,
        label_ids: list[str] | None = None,
    ) -> tuple[str, datetime]:
        """Set up Gmail push notifications for a user.

        Creates a watch on the user's mailbox so Gmail sends
        notifications via Pub/Sub when changes occur.

        Args:
            user_id: User to set up watch for.
            topic_name: Pub/Sub topic name (projects/{project}/topics/{topic}).
            label_ids: Labels to watch (default: INBOX).

        Returns:
            Tuple of (history_id, expiration datetime).

        Raises:
            ConnectionError: If not connected or no valid token.
            SyncError: If watch setup fails.
        """

        # Get valid token
        try:
            access_token = await self._auth_service.get_valid_token(user_id, "google")
        except Exception as e:
            raise ConnectionError(
                f"Not connected to Gmail: {e}",
                provider=ProviderType.GMAIL,
            ) from e

        try:
            async with GmailClient(access_token) as client:
                history_id, expiration_ms = await client.watch(
                    topic_name=topic_name,
                    label_ids=label_ids,
                )

            # Convert expiration from milliseconds to datetime
            expiration = datetime.fromtimestamp(expiration_ms / 1000, tz=UTC)

            # Store watch subscription if repo is available
            if hasattr(self, "_watch_repo") and self._watch_repo is not None:
                await self._watch_repo.activate(
                    user_id=user_id,
                    history_id=history_id,
                    expiration=expiration,
                    topic_name=topic_name,
                    label_ids=label_ids,
                )

            await logger.ainfo(
                "gmail_watch_setup",
                user_id=str(user_id),
                history_id=history_id,
                expiration=str(expiration),
            )

            return history_id, expiration

        except GmailApiError as e:
            await logger.aerror(
                "gmail_watch_failed",
                user_id=str(user_id),
                error=str(e),
            )
            raise SyncError(
                f"Failed to set up Gmail watch: {e}",
                provider=ProviderType.GMAIL,
            ) from e

    async def remove_watch(self, user_id: UUID) -> bool:
        """Remove Gmail push notifications for a user.

        Stops the current watch on the user's mailbox.

        Args:
            user_id: User to remove watch for.

        Returns:
            True if watch was removed, False if there was no watch.

        Raises:
            ConnectionError: If not connected or no valid token.
        """

        # Get valid token
        try:
            access_token = await self._auth_service.get_valid_token(user_id, "google")
        except Exception as e:
            raise ConnectionError(
                f"Not connected to Gmail: {e}",
                provider=ProviderType.GMAIL,
            ) from e

        try:
            async with GmailClient(access_token) as client:
                await client.stop_watch()

            # Deactivate watch subscription if repo is available
            if hasattr(self, "_watch_repo") and self._watch_repo is not None:
                await self._watch_repo.deactivate(user_id)

            await logger.ainfo("gmail_watch_removed", user_id=str(user_id))
            return True

        except GmailApiError as e:
            await logger.awarning(
                "gmail_watch_remove_failed",
                user_id=str(user_id),
                error=str(e),
            )
            return False

    async def renew_watch(
        self,
        user_id: UUID,
        topic_name: str,
        label_ids: list[str] | None = None,
    ) -> tuple[str, datetime]:
        """Renew an expiring watch subscription.

        This is equivalent to setup_watch but with logging specific to renewal.

        Args:
            user_id: User to renew watch for.
            topic_name: Pub/Sub topic name.
            label_ids: Labels to watch.

        Returns:
            Tuple of (history_id, new expiration datetime).
        """
        await logger.ainfo("gmail_watch_renewing", user_id=str(user_id))
        return await self.setup_watch(user_id, topic_name, label_ids)
