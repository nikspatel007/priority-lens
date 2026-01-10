"""LiveKit service for token generation."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from priority_lens.api.config import APIConfig

logger = structlog.get_logger(__name__)


class LiveKitNotConfiguredError(Exception):
    """Raised when LiveKit credentials are not configured."""

    pass


class LiveKitService:
    """Service for LiveKit token generation and room management."""

    ROOM_PREFIX = "pl-thread"

    def __init__(self, config: APIConfig) -> None:
        """Initialize LiveKit service.

        Args:
            config: API configuration with LiveKit credentials.
        """
        self._config = config
        self._api_key = config.livekit_api_key
        self._api_secret = config.livekit_api_secret
        self._url = config.livekit_url

    @property
    def is_configured(self) -> bool:
        """Check if LiveKit is properly configured."""
        return self._config.has_livekit

    def get_room_name(self, thread_id: UUID) -> str:
        """Generate deterministic room name from thread ID.

        Args:
            thread_id: Thread UUID.

        Returns:
            Room name in format: pl-thread-{thread_id}
        """
        return f"{self.ROOM_PREFIX}-{thread_id}"

    def create_token(
        self,
        thread_id: UUID,
        session_id: UUID,
        participant_identity: str,
        participant_name: str = "user",
        ttl_seconds: int = 120,
    ) -> str:
        """Create a LiveKit access token.

        Args:
            thread_id: Thread UUID for the room.
            session_id: Session UUID for metadata.
            participant_identity: Unique identity for the participant.
            participant_name: Display name for the participant.
            ttl_seconds: Token time-to-live in seconds.

        Returns:
            JWT access token string.

        Raises:
            LiveKitNotConfiguredError: If LiveKit credentials are not configured.
        """
        if not self.is_configured:
            raise LiveKitNotConfiguredError("LiveKit API key and secret must be configured")

        # Import here to avoid requiring livekit-api when not using voice features
        try:
            from livekit import api
        except ImportError as e:
            raise LiveKitNotConfiguredError(
                "livekit-api package is required. Install with: pip install livekit-api"
            ) from e

        room_name = self.get_room_name(thread_id)

        # Create access token with grants
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(participant_identity)
            .with_name(participant_name)
            .with_ttl(timedelta(seconds=ttl_seconds))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
            .with_metadata(f'{{"session_id": "{session_id}"}}')
            .to_jwt()
        )

        logger.info(
            "livekit_token_created",
            thread_id=str(thread_id),
            session_id=str(session_id),
            room_name=room_name,
            ttl_seconds=ttl_seconds,
        )

        return token

    def get_server_url(self) -> str | None:
        """Get the LiveKit server URL.

        Returns:
            LiveKit server URL or None if not configured.
        """
        return self._url
