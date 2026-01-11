"""Email provider connection endpoints."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.api.auth.dependencies import CurrentUserOrApiKey
from priority_lens.providers import (
    ConnectionService,
    ConnectionState,
    ProviderNotFoundError,
    ProviderType,
)
from priority_lens.schemas.sync import SyncStatusApiResponse
from priority_lens.services.background_sync import BackgroundSyncRunner

if TYPE_CHECKING:
    from priority_lens.api.auth.clerk import ClerkUser

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/connections", tags=["connections"])
logger = structlog.get_logger(__name__)

# Dependency for connection service - will be overridden in app setup
_connection_service: ConnectionService | None = None

# Database session factory - will be set during app initialization
_session_factory: SessionFactory | None = None


def set_session_factory(factory: SessionFactory | None) -> None:
    """Set the session factory for building per-request services.

    Args:
        factory: AsyncSession factory to use.
    """
    global _session_factory
    _session_factory = factory


def _build_default_connection_service(session: AsyncSession) -> ConnectionService:
    """Build the default ConnectionService backed by GmailProvider.

    Args:
        session: Active DB session for this request.

    Returns:
        ConnectionService configured with available providers.

    Raises:
        HTTPException: If Google OAuth env vars are missing.
    """
    from priority_lens.auth.google import GoogleOAuth
    from priority_lens.providers import GmailProvider, ProviderRegistry
    from priority_lens.repositories.oauth_token import OAuthTokenRepository
    from priority_lens.repositories.sync_state import SyncStateRepository
    from priority_lens.services.auth_service import AuthService

    google_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    google_redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI")

    # Support local `.env` without requiring callers to export env vars.
    if not google_client_id or not google_client_secret:
        try:
            from priority_lens.core.config import Config

            cfg = Config.from_env(Path(".env"))
            google_client_id = google_client_id or cfg.google_client_id
            google_client_secret = google_client_secret or cfg.google_client_secret
            google_redirect_uri = google_redirect_uri or cfg.google_redirect_uri
        except Exception:
            # Fall back to env-only configuration.
            pass

    google_redirect_uri = (
        google_redirect_uri or "http://localhost:8000/api/v1/connections/gmail/callback"
    )

    if not google_client_id or not google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )

    token_repo = OAuthTokenRepository(session)
    sync_repo = SyncStateRepository(session)

    oauth = GoogleOAuth(
        client_id=google_client_id,
        client_secret=google_client_secret,
        redirect_uri=google_redirect_uri,
    )
    auth_service = AuthService(oauth, token_repo)

    registry = ProviderRegistry()
    registry.register(GmailProvider(auth_service, token_repo, sync_repo))

    return ConnectionService(registry)


async def get_connection_service() -> AsyncGenerator[ConnectionService, None]:
    """Get the connection service instance.

    Returns:
        ConnectionService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    if _connection_service is None:
        if _session_factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Connection service not configured",
            )
        async with _session_factory() as session:
            yield _build_default_connection_service(session)
        return

    yield _connection_service


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for user provisioning.

    Yields:
        AsyncSession for database operations.

    Raises:
        HTTPException: If session factory not configured.
    """
    if _session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database session not configured",
        )
    async with _session_factory() as session:
        yield session


def set_connection_service(service: ConnectionService | None) -> None:
    """Set the connection service instance.

    Args:
        service: ConnectionService to use, or None to reset.
    """
    global _connection_service
    _connection_service = service


ConnectionServiceDep = Annotated[ConnectionService, Depends(get_connection_service)]
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


class ProviderStatusResponse(BaseModel):
    """Provider connection status response."""

    provider: str = Field(description="Provider type identifier")
    state: str = Field(description="Connection state")
    is_connected: bool = Field(description="Whether provider is connected")
    email: str | None = Field(default=None, description="Connected email address")
    connected_at: str | None = Field(default=None, description="When connection was established")
    last_sync: str | None = Field(default=None, description="When last sync occurred")
    error: str | None = Field(default=None, description="Error message if any")


class AllConnectionsResponse(BaseModel):
    """Response containing all provider statuses."""

    providers: list[ProviderStatusResponse] = Field(description="Status of all providers")
    connected_count: int = Field(description="Number of connected providers")


class AuthUrlResponse(BaseModel):
    """OAuth authorization URL response."""

    auth_url: str = Field(description="URL to redirect user for authorization")
    provider: str = Field(description="Provider being authorized")


class ConnectResponse(BaseModel):
    """Connection completion response."""

    provider: str = Field(description="Connected provider")
    state: str = Field(description="Connection state")
    message: str = Field(description="Status message")
    sync_started: bool = Field(
        default=False,
        description="Whether initial sync was started for first-time users",
    )


class DisconnectResponse(BaseModel):
    """Disconnection response."""

    provider: str = Field(description="Disconnected provider")
    disconnected: bool = Field(description="Whether disconnection was successful")
    message: str = Field(description="Status message")


class SyncProgressResponse(BaseModel):
    """Sync progress response."""

    provider: str = Field(description="Provider being synced")
    in_progress: bool = Field(description="Whether sync is in progress")
    processed: int | None = Field(default=None, description="Messages processed")
    total: int | None = Field(default=None, description="Total messages to process")
    phase: str | None = Field(default=None, description="Current sync phase")


def _get_user_uuid(user: ClerkUser) -> UUID:
    """Convert Clerk user ID to UUID.

    Args:
        user: Clerk user.

    Returns:
        UUID derived from user ID.

    Note:
        Uses UUID5 with DNS namespace to create deterministic UUID from Clerk ID.
    """
    import uuid

    return uuid.uuid5(uuid.NAMESPACE_DNS, user.id)


async def _ensure_user_exists(user: ClerkUser, session: AsyncSession) -> UUID:
    """Ensure the Clerk user exists in org_users table.

    Creates a default organization and user if they don't exist.
    This enables OAuth token storage for new users.

    Args:
        user: Clerk user from JWT.
        session: Database session.

    Returns:
        User UUID (either existing or newly created).
    """
    import uuid as uuid_module

    from sqlalchemy import select

    from priority_lens.models.org_user import OrgUser
    from priority_lens.models.organization import Organization

    user_uuid = uuid_module.uuid5(uuid_module.NAMESPACE_DNS, user.id)

    # Check if user already exists
    result = await session.execute(select(OrgUser).where(OrgUser.id == user_uuid))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        return user_uuid

    # User doesn't exist - create with default org
    # First, get or create default organization
    default_org_id = uuid_module.UUID("00000000-0000-0000-0000-000000000001")
    org_result = await session.execute(
        select(Organization).where(Organization.id == default_org_id)
    )
    org = org_result.scalar_one_or_none()

    if not org:
        org = Organization(
            id=default_org_id,
            name="Default Organization",
            slug="default",
        )
        session.add(org)
        await session.flush()

    # Create the user with specific ID
    new_user = OrgUser(
        id=user_uuid,
        org_id=default_org_id,
        email=user.email or f"{user.id}@clerk.user",
        name=user.full_name,
        role="member",
    )
    session.add(new_user)
    await session.commit()

    await logger.ainfo(
        "user_auto_provisioned",
        user_id=str(user_uuid),
        clerk_id=user.id,
        email=user.email,
    )

    return user_uuid


@router.get(
    "/available",
    response_model=list[str],
    summary="List available providers",
    description="List all available (registered) email providers.",
)
async def list_available_providers(
    service: ConnectionServiceDep,
) -> list[str]:
    """List all available providers.

    Returns list of provider identifiers that can be connected.

    Args:
        service: Connection service.

    Returns:
        List of provider type strings.
    """
    providers = service.list_available_providers()
    return [p.value for p in providers]


@router.get(
    "",
    response_model=AllConnectionsResponse,
    summary="List all connections",
    description="Get connection status for all available email providers.",
)
async def list_connections(
    user: CurrentUserOrApiKey,
    service: ConnectionServiceDep,
) -> AllConnectionsResponse:
    """Get connection status for all providers.

    Returns status of all registered providers for the authenticated user.

    Args:
        user: Current authenticated user.
        service: Connection service.

    Returns:
        AllConnectionsResponse with status of each provider.
    """
    user_uuid = _get_user_uuid(user)

    statuses = await service.get_all_statuses(user_uuid)

    providers = []
    connected_count = 0

    for provider_type, connection_status in statuses.items():
        if connection_status.is_connected:
            connected_count += 1

        providers.append(
            ProviderStatusResponse(
                provider=provider_type.value,
                state=connection_status.state.value,
                is_connected=connection_status.is_connected,
                email=connection_status.email,
                connected_at=(
                    connection_status.connected_at.isoformat()
                    if connection_status.connected_at
                    else None
                ),
                last_sync=(
                    connection_status.last_sync.isoformat() if connection_status.last_sync else None
                ),
                error=connection_status.error,
            )
        )

    await logger.ainfo(
        "connections_listed",
        user_id=user.id,
        connected_count=connected_count,
    )

    return AllConnectionsResponse(
        providers=providers,
        connected_count=connected_count,
    )


@router.get(
    "/{provider}",
    response_model=ProviderStatusResponse,
    summary="Get provider status",
    description="Get connection status for a specific provider.",
)
async def get_connection_status(
    provider: str,
    user: CurrentUserOrApiKey,
    service: ConnectionServiceDep,
) -> ProviderStatusResponse:
    """Get connection status for a specific provider.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.

    Returns:
        ProviderStatusResponse with provider status.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    try:
        connection_status = await service.get_status(user_uuid, provider_type)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    return ProviderStatusResponse(
        provider=provider_type.value,
        state=connection_status.state.value,
        is_connected=connection_status.is_connected,
        email=connection_status.email,
        connected_at=(
            connection_status.connected_at.isoformat() if connection_status.connected_at else None
        ),
        last_sync=(
            connection_status.last_sync.isoformat() if connection_status.last_sync else None
        ),
        error=connection_status.error,
    )


@router.post(
    "/{provider}/connect",
    response_model=AuthUrlResponse,
    summary="Start OAuth flow",
    description="Get authorization URL to connect a provider.",
)
async def start_connection(
    provider: str,
    user: CurrentUserOrApiKey,
    service: ConnectionServiceDep,
    state: Annotated[str | None, Query(description="CSRF state parameter")] = None,
) -> AuthUrlResponse:
    """Start OAuth authorization flow.

    Returns URL to redirect user to for provider authorization.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.
        state: Optional CSRF protection state.

    Returns:
        AuthUrlResponse with authorization URL.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    try:
        auth_url = await service.get_auth_url(provider_type, state=state)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    await logger.ainfo(
        "auth_flow_started",
        user_id=user.id,
        provider=provider,
    )

    return AuthUrlResponse(
        auth_url=auth_url,
        provider=provider,
    )


@router.post(
    "/{provider}/callback",
    response_model=ConnectResponse,
    summary="Complete OAuth flow",
    description="Complete OAuth authorization with the callback code.",
)
async def complete_connection(
    provider: str,
    code: Annotated[str, Query(description="Authorization code from OAuth callback")],
    user: CurrentUserOrApiKey,
    service: ConnectionServiceDep,
    runner: BackgroundSyncRunnerDep,
    db: DbSessionDep,
    mobile: Annotated[
        bool, Query(description="Set to true for mobile OAuth (serverAuthCode)")
    ] = False,
) -> ConnectResponse:
    """Complete OAuth authorization flow.

    Exchanges authorization code for tokens and establishes connection.
    For first-time users, triggers background sync of emails.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        code: Authorization code from OAuth callback.
        user: Current authenticated user.
        service: Connection service.
        runner: Background sync runner.
        db: Database session for user provisioning.

    Returns:
        ConnectResponse with connection result and sync status.

    Raises:
        HTTPException: If authorization fails.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    # Ensure user exists in org_users before OAuth token storage
    user_uuid = await _ensure_user_exists(user, db)

    try:
        connection_status = await service.complete_auth(
            user_uuid, provider_type, code, from_mobile=mobile
        )
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None
    except Exception as e:
        await logger.aerror(
            "auth_completion_failed",
            user_id=user.id,
            provider=provider,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authorization failed: {e}",
        ) from None

    await logger.ainfo(
        "connection_established",
        user_id=user.id,
        provider=provider,
        state=connection_status.state.value,
    )

    # Start initial sync for first-time users
    sync_started = False
    if connection_status.state == ConnectionState.CONNECTED:
        sync_started = await runner.start_if_needed(user_uuid)
        if sync_started:
            await logger.ainfo(
                "initial_sync_started",
                user_id=user.id,
                provider=provider,
            )

    message = (
        "Successfully connected"
        if connection_status.state == ConnectionState.CONNECTED
        else f"Connection state: {connection_status.state.value}"
    )

    return ConnectResponse(
        provider=provider,
        state=connection_status.state.value,
        message=message,
        sync_started=sync_started,
    )


@router.delete(
    "/{provider}",
    response_model=DisconnectResponse,
    summary="Disconnect provider",
    description="Disconnect and revoke access for a provider.",
)
async def disconnect_provider(
    provider: str,
    user: CurrentUserOrApiKey,
    service: ConnectionServiceDep,
) -> DisconnectResponse:
    """Disconnect from a provider.

    Revokes OAuth tokens and removes connection.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.

    Returns:
        DisconnectResponse with disconnection result.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    try:
        disconnected = await service.disconnect(user_uuid, provider_type)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    message = "Disconnected successfully" if disconnected else "Was not connected"

    await logger.ainfo(
        "provider_disconnected",
        user_id=user.id,
        provider=provider,
        disconnected=disconnected,
    )

    return DisconnectResponse(
        provider=provider,
        disconnected=disconnected,
        message=message,
    )


@router.get(
    "/{provider}/sync/progress",
    response_model=SyncProgressResponse,
    summary="Get sync progress",
    description="Get current sync progress for a provider.",
)
async def get_sync_progress(
    provider: str,
    user: CurrentUserOrApiKey,
    service: ConnectionServiceDep,
) -> SyncProgressResponse:
    """Get sync progress for a provider.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        service: Connection service.

    Returns:
        SyncProgressResponse with sync progress.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    try:
        progress = await service.get_sync_progress(user_uuid, provider_type)
    except ProviderNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not available: {provider}",
        ) from None

    if progress is None:
        return SyncProgressResponse(
            provider=provider,
            in_progress=False,
        )

    return SyncProgressResponse(
        provider=provider,
        in_progress=True,
        processed=progress.processed,
        total=progress.total,
        phase=progress.current_phase,
    )


# Global background sync runner - set during app initialization
_background_sync_runner: BackgroundSyncRunner | None = None


async def get_background_sync_runner() -> AsyncGenerator[BackgroundSyncRunner, None]:
    """Get the background sync runner instance.

    Returns:
        BackgroundSyncRunner instance.

    Raises:
        HTTPException: If runner is not configured.
    """
    if _background_sync_runner is None:
        if _session_factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Background sync runner not configured",
            )
        async with _session_factory() as session:
            from priority_lens.repositories.sync_state import SyncStateRepository

            yield BackgroundSyncRunner(SyncStateRepository(session))
        return

    yield _background_sync_runner


def set_background_sync_runner(runner: BackgroundSyncRunner | None) -> None:
    """Set the background sync runner instance.

    Args:
        runner: BackgroundSyncRunner to use, or None to reset.
    """
    global _background_sync_runner
    _background_sync_runner = runner


BackgroundSyncRunnerDep = Annotated["BackgroundSyncRunner", Depends(get_background_sync_runner)]


@router.get(
    "/{provider}/sync/status",
    response_model=SyncStatusApiResponse,
    summary="Get sync status for mobile app",
    description="Get current sync status for polling by mobile app.",
)
async def get_sync_status_api(
    provider: str,
    user: CurrentUserOrApiKey,
    runner: BackgroundSyncRunnerDep,
) -> SyncStatusApiResponse:
    """Get sync status for mobile app polling.

    Returns status in mobile-friendly format with progress percentage.

    Args:
        provider: Provider identifier (e.g., 'gmail').
        user: Current authenticated user.
        runner: Background sync runner.

    Returns:
        SyncStatusApiResponse with current sync state.

    Raises:
        HTTPException: If provider is not found.
    """
    try:
        ProviderType(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}",
        ) from None

    user_uuid = _get_user_uuid(user)

    sync_status = await runner.get_status(user_uuid)

    return SyncStatusApiResponse(
        status=sync_status.status,
        emails_synced=sync_status.emails_synced,
        total_emails=sync_status.total_emails,
        progress=sync_status.progress,
        error=sync_status.error,
    )
