"""Tests for webhook endpoints."""

from __future__ import annotations

import base64
import json
from unittest import mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from priority_lens.api.routes.webhooks import router, set_push_service
from priority_lens.services.push_notification import NotificationResult


@pytest.fixture
def mock_push_service() -> mock.MagicMock:
    """Create mock push notification service."""
    service = mock.MagicMock()
    service.handle_notification = mock.AsyncMock(
        return_value=NotificationResult(success=True, emails_synced=0)
    )
    return service


@pytest.fixture
def app(mock_push_service: mock.MagicMock) -> FastAPI:
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router)

    set_push_service(mock_push_service)

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestGmailWebhook:
    """Tests for /webhooks/gmail endpoint."""

    def test_webhook_success(self, client: TestClient, mock_push_service: mock.MagicMock) -> None:
        """Test successful webhook processing."""
        notification_data = {
            "emailAddress": "test@example.com",
            "historyId": "12345",
        }
        encoded = base64.b64encode(json.dumps(notification_data).encode()).decode()

        response = client.post(
            "/webhooks/gmail",
            json={
                "message": {
                    "data": encoded,
                    "message_id": "msg-123",
                },
                "subscription": "projects/test/subscriptions/gmail",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["processed"] is True
        mock_push_service.handle_notification.assert_called_once_with(encoded)

    def test_webhook_skipped(self, client: TestClient, mock_push_service: mock.MagicMock) -> None:
        """Test webhook when notification is skipped (duplicate)."""
        mock_push_service.handle_notification.return_value = NotificationResult(
            success=True, skipped=True
        )

        notification_data = {
            "emailAddress": "test@example.com",
            "historyId": "12345",
        }
        encoded = base64.b64encode(json.dumps(notification_data).encode()).decode()

        response = client.post(
            "/webhooks/gmail",
            json={"message": {"data": encoded}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["processed"] is False
        assert data["skipped"] is True

    def test_webhook_with_emails_synced(
        self, client: TestClient, mock_push_service: mock.MagicMock
    ) -> None:
        """Test webhook response includes emails_synced."""
        user_id = uuid4()
        mock_push_service.handle_notification.return_value = NotificationResult(
            success=True, user_id=user_id, emails_synced=5
        )

        notification_data = {
            "emailAddress": "test@example.com",
            "historyId": "12345",
        }
        encoded = base64.b64encode(json.dumps(notification_data).encode()).decode()

        response = client.post(
            "/webhooks/gmail",
            json={"message": {"data": encoded}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["emails_synced"] == 5

    def test_webhook_error_still_returns_200(
        self, client: TestClient, mock_push_service: mock.MagicMock
    ) -> None:
        """Test webhook returns 200 even on processing error (to prevent retries)."""
        mock_push_service.handle_notification.return_value = NotificationResult(
            success=False, error="Processing failed"
        )

        notification_data = {
            "emailAddress": "test@example.com",
            "historyId": "12345",
        }
        encoded = base64.b64encode(json.dumps(notification_data).encode()).decode()

        response = client.post(
            "/webhooks/gmail",
            json={"message": {"data": encoded}},
        )

        # Still 200 to prevent Pub/Sub retries
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["processed"] is False

    def test_webhook_missing_message(self, client: TestClient) -> None:
        """Test webhook with missing message returns 422."""
        response = client.post("/webhooks/gmail", json={})

        assert response.status_code == 422

    def test_webhook_missing_data(self, client: TestClient) -> None:
        """Test webhook with missing data field returns 422."""
        response = client.post("/webhooks/gmail", json={"message": {}})

        assert response.status_code == 422


class TestGmailWebhookRaw:
    """Tests for /webhooks/gmail/raw endpoint."""

    def test_raw_webhook_success(
        self, client: TestClient, mock_push_service: mock.MagicMock
    ) -> None:
        """Test raw webhook endpoint."""
        notification_data = {
            "emailAddress": "test@example.com",
            "historyId": "12345",
        }
        encoded = base64.b64encode(json.dumps(notification_data).encode()).decode()

        response = client.post(
            "/webhooks/gmail/raw",
            json={"data": encoded},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        mock_push_service.handle_notification.assert_called_once_with(encoded)

    def test_raw_webhook_missing_data(self, client: TestClient) -> None:
        """Test raw webhook with missing data returns 400."""
        response = client.post("/webhooks/gmail/raw", json={})

        assert response.status_code == 400
        assert "data" in response.json()["detail"].lower()


class TestWebhookServiceDependency:
    """Tests for push service dependency."""

    def test_service_not_configured(self) -> None:
        """Test error when service not configured."""
        app = FastAPI()
        app.include_router(router)

        # Reset the global service
        set_push_service(None)  # type: ignore[arg-type]

        client = TestClient(app)

        notification_data = {
            "emailAddress": "test@example.com",
            "historyId": "12345",
        }
        encoded = base64.b64encode(json.dumps(notification_data).encode()).decode()

        response = client.post(
            "/webhooks/gmail",
            json={"message": {"data": encoded}},
        )

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
