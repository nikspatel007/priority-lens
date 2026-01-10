"""LiveKit API endpoints for token generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from priority_lens.api.auth.dependencies import CurrentUserOrApiKey
from priority_lens.api.config import APIConfig, get_api_config
from priority_lens.schemas.livekit import (
    LiveKitConfig,
    LiveKitTokenRequest,
    LiveKitTokenResponse,
)
from priority_lens.services.livekit_service import (
    LiveKitNotConfiguredError,
    LiveKitService,
)

if TYPE_CHECKING:
    from priority_lens.api.auth.clerk import ClerkUser

router = APIRouter(prefix="/livekit", tags=["livekit"])
logger = structlog.get_logger(__name__)

# Service dependency - lazily initialized
_livekit_service: LiveKitService | None = None


def get_livekit_service(
    config: Annotated[APIConfig, Depends(get_api_config)],
) -> LiveKitService:
    """Get LiveKit service instance.

    Args:
        config: API configuration.

    Returns:
        LiveKitService instance.
    """
    global _livekit_service
    if _livekit_service is None:
        _livekit_service = LiveKitService(config)
    return _livekit_service


def set_livekit_service(service: LiveKitService | None) -> None:
    """Set LiveKit service for testing.

    Args:
        service: LiveKitService instance or None to reset.
    """
    global _livekit_service
    _livekit_service = service


LiveKitServiceDep = Annotated[LiveKitService, Depends(get_livekit_service)]


def _get_user_id(user: ClerkUser) -> str:
    """Get user identity string from ClerkUser.

    Args:
        user: Authenticated user.

    Returns:
        User identity string for LiveKit.
    """
    return user.id


def _get_user_id_uuid(user: ClerkUser) -> UUID:
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


@router.get("/config", response_model=LiveKitConfig)
async def get_config(
    service: LiveKitServiceDep,
) -> LiveKitConfig:
    """Get LiveKit configuration status."""
    return LiveKitConfig(
        enabled=service.is_configured,
        url=service.get_server_url(),
    )


@router.post("/token", response_model=LiveKitTokenResponse)
async def create_token(
    data: LiveKitTokenRequest,
    user: CurrentUserOrApiKey,
    service: LiveKitServiceDep,
) -> LiveKitTokenResponse:
    """Generate a LiveKit access token for a room.

    The room name is deterministic based on the thread_id.
    Tokens are short-lived (default 120 seconds).
    """
    try:
        user_identity = _get_user_id(user)
        token = service.create_token(
            thread_id=data.thread_id,
            session_id=data.session_id,
            participant_identity=user_identity,
            participant_name=data.participant_name,
            ttl_seconds=data.ttl_seconds,
        )

        room_name = service.get_room_name(data.thread_id)
        livekit_url = service.get_server_url()

        if not livekit_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LiveKit server URL not configured",
            )

        await logger.ainfo(
            "livekit_token_issued",
            user_id=user_identity,
            thread_id=str(data.thread_id),
            session_id=str(data.session_id),
            room_name=room_name,
            ttl_seconds=data.ttl_seconds,
        )

        return LiveKitTokenResponse(
            token=token,
            room_name=room_name,
            livekit_url=livekit_url,
            expires_in=data.ttl_seconds,
        )

    except LiveKitNotConfiguredError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
