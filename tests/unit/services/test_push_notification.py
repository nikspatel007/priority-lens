"""Tests for PushNotificationService."""

from __future__ import annotations

import base64
import json
import time
from unittest import mock
from uuid import uuid4

import pytest

from priority_lens.services.push_notification import (
    InvalidNotificationError,
    NotificationData,
    NotificationDeduplicator,
    NotificationResult,
    PushNotificationService,
)


class TestNotificationDeduplicator:
    """Tests for NotificationDeduplicator."""

    def test_is_duplicate_returns_false_for_new_key(self) -> None:
        """Test new keys are not duplicates."""
        dedup = NotificationDeduplicator()

        assert dedup.is_duplicate("key1") is False

    def test_is_duplicate_returns_true_for_seen_key(self) -> None:
        """Test seen keys are duplicates."""
        dedup = NotificationDeduplicator()
        dedup.mark_processed("key1")

        assert dedup.is_duplicate("key1") is True

    def test_mark_processed_adds_key(self) -> None:
        """Test mark_processed adds key to cache."""
        dedup = NotificationDeduplicator()

        dedup.mark_processed("key1")

        assert dedup.is_duplicate("key1") is True

    def test_expired_entries_are_removed(self) -> None:
        """Test expired entries are cleaned up."""
        dedup = NotificationDeduplicator(ttl_seconds=1)
        dedup.mark_processed("key1")

        # Wait for TTL to expire
        time.sleep(1.1)

        assert dedup.is_duplicate("key1") is False

    def test_max_entries_eviction(self) -> None:
        """Test oldest entries are evicted at capacity."""
        dedup = NotificationDeduplicator(max_entries=2)

        dedup.mark_processed("key1")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        dedup.mark_processed("key2")
        time.sleep(0.01)
        dedup.mark_processed("key3")

        # key1 should be evicted (oldest)
        assert dedup.is_duplicate("key1") is False
        assert dedup.is_duplicate("key2") is True
        assert dedup.is_duplicate("key3") is True


class TestNotificationData:
    """Tests for NotificationData dataclass."""

    def test_create_notification_data(self) -> None:
        """Test creating notification data."""
        data = NotificationData(
            email_address="test@example.com",
            history_id="12345",
            raw_data={"key": "value"},
        )

        assert data.email_address == "test@example.com"
        assert data.history_id == "12345"
        assert data.raw_data == {"key": "value"}


class TestNotificationResult:
    """Tests for NotificationResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating successful result."""
        user_id = uuid4()
        result = NotificationResult(
            success=True,
            user_id=user_id,
            emails_synced=5,
        )

        assert result.success is True
        assert result.user_id == user_id
        assert result.emails_synced == 5
        assert result.skipped is False
        assert result.error is None

    def test_create_skipped_result(self) -> None:
        """Test creating skipped result."""
        result = NotificationResult(success=True, skipped=True)

        assert result.success is True
        assert result.skipped is True

    def test_create_error_result(self) -> None:
        """Test creating error result."""
        result = NotificationResult(success=False, error="Something failed")

        assert result.success is False
        assert result.error == "Something failed"


class TestPushNotificationServiceSetSyncService:
    """Tests for PushNotificationService.set_sync_service."""

    def test_set_sync_service(self) -> None:
        """Test setting the sync service."""
        user_repo = mock.MagicMock()
        sync_repo = mock.MagicMock()
        watch_repo = mock.MagicMock()
        service = PushNotificationService(user_repo, sync_repo, watch_repo)

        mock_sync_service = mock.MagicMock()
        service.set_sync_service(mock_sync_service)

        assert service.sync_service == mock_sync_service


class TestPushNotificationServiceParseNotification:
    """Tests for PushNotificationService.parse_notification."""

    def _create_service(self) -> PushNotificationService:
        """Create a service with mocked dependencies."""
        user_repo = mock.MagicMock()
        sync_repo = mock.MagicMock()
        watch_repo = mock.MagicMock()
        return PushNotificationService(user_repo, sync_repo, watch_repo)

    def test_parse_valid_notification(self) -> None:
        """Test parsing valid notification data."""
        service = self._create_service()
        payload = {
            "emailAddress": "test@example.com",
            "historyId": "12345",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()

        result = service.parse_notification(encoded)

        assert result.email_address == "test@example.com"
        assert result.history_id == "12345"

    def test_parse_invalid_base64(self) -> None:
        """Test parsing invalid base64 raises error."""
        service = self._create_service()

        with pytest.raises(InvalidNotificationError, match="decode"):
            service.parse_notification("not-base64!")

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON raises error."""
        service = self._create_service()
        encoded = base64.b64encode(b"not json").decode()

        with pytest.raises(InvalidNotificationError, match="decode"):
            service.parse_notification(encoded)

    def test_parse_missing_email(self) -> None:
        """Test parsing data without email raises error."""
        service = self._create_service()
        payload = {"historyId": "12345"}
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()

        with pytest.raises(InvalidNotificationError, match="emailAddress"):
            service.parse_notification(encoded)

    def test_parse_missing_history_id(self) -> None:
        """Test parsing data without historyId raises error."""
        service = self._create_service()
        payload = {"emailAddress": "test@example.com"}
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()

        with pytest.raises(InvalidNotificationError, match="historyId"):
            service.parse_notification(encoded)


class TestPushNotificationServiceHandleNotification:
    """Tests for PushNotificationService.handle_notification."""

    def _create_encoded_notification(
        self, email: str = "test@example.com", history_id: str = "12345"
    ) -> str:
        """Create an encoded notification payload."""
        payload = {"emailAddress": email, "historyId": history_id}
        return base64.b64encode(json.dumps(payload).encode()).decode()

    @pytest.fixture
    def user_repo(self) -> mock.MagicMock:
        """Create mock user repository."""
        repo = mock.MagicMock()
        repo.find_by_email = mock.AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def sync_repo(self) -> mock.MagicMock:
        """Create mock sync repository."""
        return mock.MagicMock()

    @pytest.fixture
    def watch_repo(self) -> mock.MagicMock:
        """Create mock watch repository."""
        repo = mock.MagicMock()
        repo.record_notification = mock.AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_handle_invalid_notification(
        self,
        user_repo: mock.MagicMock,
        sync_repo: mock.MagicMock,
        watch_repo: mock.MagicMock,
    ) -> None:
        """Test handling invalid notification returns error."""
        service = PushNotificationService(user_repo, sync_repo, watch_repo)

        result = await service.handle_notification("invalid-data")

        assert result.success is False
        assert result.error is not None
        assert "decode" in result.error

    @pytest.mark.asyncio
    async def test_handle_user_not_found(
        self,
        user_repo: mock.MagicMock,
        sync_repo: mock.MagicMock,
        watch_repo: mock.MagicMock,
    ) -> None:
        """Test handling notification when user not found."""
        user_repo.find_by_email.return_value = None
        service = PushNotificationService(user_repo, sync_repo, watch_repo)
        encoded = self._create_encoded_notification()

        result = await service.handle_notification(encoded)

        assert result.success is False
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_handle_duplicate_notification(
        self,
        user_repo: mock.MagicMock,
        sync_repo: mock.MagicMock,
        watch_repo: mock.MagicMock,
    ) -> None:
        """Test handling duplicate notification is skipped."""
        mock_user = mock.MagicMock()
        mock_user.id = uuid4()
        user_repo.find_by_email.return_value = mock_user
        service = PushNotificationService(user_repo, sync_repo, watch_repo)
        encoded = self._create_encoded_notification()

        # First call - should process
        result1 = await service.handle_notification(encoded)
        assert result1.success is True
        assert result1.skipped is False

        # Second call - should skip
        result2 = await service.handle_notification(encoded)
        assert result2.success is True
        assert result2.skipped is True

    @pytest.mark.asyncio
    async def test_handle_success_without_sync_service(
        self,
        user_repo: mock.MagicMock,
        sync_repo: mock.MagicMock,
        watch_repo: mock.MagicMock,
    ) -> None:
        """Test handling notification without sync service."""
        mock_user = mock.MagicMock()
        mock_user.id = uuid4()
        user_repo.find_by_email.return_value = mock_user
        service = PushNotificationService(user_repo, sync_repo, watch_repo)
        encoded = self._create_encoded_notification()

        result = await service.handle_notification(encoded)

        assert result.success is True
        assert result.user_id == mock_user.id
        assert result.emails_synced == 0
        watch_repo.record_notification.assert_called_once_with(mock_user.id)

    @pytest.mark.asyncio
    async def test_handle_success_with_sync_service(
        self,
        user_repo: mock.MagicMock,
        sync_repo: mock.MagicMock,
        watch_repo: mock.MagicMock,
    ) -> None:
        """Test handling notification with sync service triggers sync."""
        mock_user = mock.MagicMock()
        mock_user.id = uuid4()
        user_repo.find_by_email.return_value = mock_user

        mock_sync_service = mock.MagicMock()
        mock_sync_result = mock.MagicMock()
        mock_sync_result.emails_synced = 5
        mock_sync_service.incremental_sync = mock.AsyncMock(return_value=(mock_sync_result, []))

        service = PushNotificationService(user_repo, sync_repo, watch_repo, mock_sync_service)
        encoded = self._create_encoded_notification()

        result = await service.handle_notification(encoded)

        assert result.success is True
        assert result.emails_synced == 5
        mock_sync_service.incremental_sync.assert_called_once_with(mock_user.id)

    @pytest.mark.asyncio
    async def test_handle_sync_failure(
        self,
        user_repo: mock.MagicMock,
        sync_repo: mock.MagicMock,
        watch_repo: mock.MagicMock,
    ) -> None:
        """Test handling notification when sync fails."""
        mock_user = mock.MagicMock()
        mock_user.id = uuid4()
        user_repo.find_by_email.return_value = mock_user

        mock_sync_service = mock.MagicMock()
        mock_sync_service.incremental_sync = mock.AsyncMock(side_effect=Exception("Sync error"))

        service = PushNotificationService(user_repo, sync_repo, watch_repo, mock_sync_service)
        encoded = self._create_encoded_notification()

        result = await service.handle_notification(encoded)

        assert result.success is False
        assert result.user_id == mock_user.id
        assert "Sync failed" in (result.error or "")


class TestPushNotificationServiceHandleBatch:
    """Tests for PushNotificationService.handle_notification_batch."""

    @pytest.mark.asyncio
    async def test_handle_batch(self) -> None:
        """Test handling batch of notifications."""
        user_repo = mock.MagicMock()
        sync_repo = mock.MagicMock()
        watch_repo = mock.MagicMock()
        watch_repo.record_notification = mock.AsyncMock()

        mock_user = mock.MagicMock()
        mock_user.id = uuid4()
        user_repo.find_by_email = mock.AsyncMock(return_value=mock_user)

        service = PushNotificationService(user_repo, sync_repo, watch_repo)

        # Create two different notifications
        notification1 = base64.b64encode(
            json.dumps({"emailAddress": "test1@example.com", "historyId": "111"}).encode()
        ).decode()
        notification2 = base64.b64encode(
            json.dumps({"emailAddress": "test2@example.com", "historyId": "222"}).encode()
        ).decode()

        results = await service.handle_notification_batch([notification1, notification2])

        assert len(results) == 2
        assert all(r.success for r in results)
