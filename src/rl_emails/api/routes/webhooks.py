"""Webhook endpoints for external notifications."""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from rl_emails.services.push_notification import PushNotificationService

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
logger = structlog.get_logger(__name__)

# Dependency for push notification service - will be overridden in app setup
_push_service: PushNotificationService | None = None


def get_push_service() -> PushNotificationService:
    """Get the push notification service instance.

    Returns:
        PushNotificationService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    if _push_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Push notification service not configured",
        )
    return _push_service


def set_push_service(service: PushNotificationService) -> None:
    """Set the push notification service instance.

    Args:
        service: PushNotificationService to use.
    """
    global _push_service
    _push_service = service


PushServiceDep = Annotated[PushNotificationService, Depends(get_push_service)]


class PubSubMessage(BaseModel):
    """Google Cloud Pub/Sub message format."""

    data: str = Field(description="Base64-encoded message data")
    message_id: str | None = Field(default=None, description="Pub/Sub message ID")
    publish_time: str | None = Field(default=None, description="When message was published")


class PubSubPushRequest(BaseModel):
    """Google Cloud Pub/Sub push request format."""

    message: PubSubMessage = Field(description="The Pub/Sub message")
    subscription: str | None = Field(default=None, description="Subscription name")


class WebhookResponse(BaseModel):
    """Response for webhook endpoints."""

    status: str = Field(description="Processing status")
    processed: bool = Field(description="Whether notification was processed")
    skipped: bool = Field(default=False, description="Whether notification was skipped")
    emails_synced: int = Field(default=0, description="Number of emails synced")


@router.post(
    "/gmail",
    response_model=WebhookResponse,
    summary="Gmail push notification webhook",
    description="Receives push notifications from Gmail via Google Cloud Pub/Sub.",
    responses={
        200: {"description": "Notification processed successfully"},
        400: {"description": "Invalid notification data"},
        503: {"description": "Service not configured"},
    },
)
async def gmail_webhook(
    request: Request,
    push_request: PubSubPushRequest,
    service: PushServiceDep,
) -> WebhookResponse:
    """Handle Gmail push notification webhook.

    This endpoint receives notifications from Google Cloud Pub/Sub
    when Gmail mailbox changes occur. It triggers an incremental
    sync to fetch new emails.

    The request must be from Google Cloud Pub/Sub push delivery.

    Args:
        request: FastAPI request object.
        push_request: Pub/Sub push request body.
        service: Push notification service.

    Returns:
        WebhookResponse with processing status.

    Note:
        Returns 200 even on processing errors to prevent Pub/Sub retries
        for unrecoverable errors. Errors are logged for monitoring.
    """
    # Log incoming request
    await logger.ainfo(
        "gmail_webhook_received",
        message_id=push_request.message.message_id,
        subscription=push_request.subscription,
    )

    # Process the notification
    result = await service.handle_notification(push_request.message.data)

    if not result.success and not result.skipped:
        # Log error but still return 200 to prevent retries
        await logger.aerror(
            "gmail_webhook_processing_failed",
            error=result.error,
            message_id=push_request.message.message_id,
        )

    return WebhookResponse(
        status="ok" if result.success else "error",
        processed=result.success and not result.skipped,
        skipped=result.skipped,
        emails_synced=result.emails_synced,
    )


@router.post(
    "/gmail/raw",
    response_model=WebhookResponse,
    summary="Gmail raw notification webhook",
    description="Receives raw notification data (for testing).",
)
async def gmail_webhook_raw(
    request: Request,
    service: PushServiceDep,
) -> WebhookResponse:
    """Handle raw Gmail notification (for testing).

    This endpoint accepts raw JSON body for easier testing.
    The body should contain a 'data' field with base64-encoded
    notification data.

    Args:
        request: FastAPI request object.
        service: Push notification service.

    Returns:
        WebhookResponse with processing status.
    """
    body = await request.json()

    message_data = body.get("data")
    if not message_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'data' field in request body",
        )

    result = await service.handle_notification(message_data)

    return WebhookResponse(
        status="ok" if result.success else "error",
        processed=result.success and not result.skipped,
        skipped=result.skipped,
        emails_synced=result.emails_synced,
    )
