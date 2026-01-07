"""Tests for Gmail sync service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from rl_emails.integrations.gmail.client import GmailApiError
from rl_emails.integrations.gmail.models import GmailMessageRef
from rl_emails.services.sync_service import SyncResult, SyncService


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful sync result."""
        result = SyncResult(
            success=True,
            emails_synced=100,
            history_id="12345",
        )

        assert result.success is True
        assert result.emails_synced == 100
        assert result.history_id == "12345"
        assert result.error is None

    def test_create_error_result(self) -> None:
        """Test creating an error sync result."""
        result = SyncResult(
            success=False,
            emails_synced=0,
            error="API error occurred",
        )

        assert result.success is False
        assert result.emails_synced == 0
        assert result.error == "API error occurred"


class TestSyncServiceInit:
    """Tests for SyncService initialization."""

    def test_init_stores_dependencies(self) -> None:
        """Test that init stores client and repo."""
        gmail_client = MagicMock()
        sync_repo = MagicMock()

        service = SyncService(
            gmail_client=gmail_client,
            sync_repo=sync_repo,
        )

        assert service.gmail_client == gmail_client
        assert service.sync_repo == sync_repo


class TestSyncServiceInitialSync:
    """Tests for SyncService.initial_sync method."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        repo = MagicMock()
        repo.start_sync = AsyncMock()
        repo.complete_sync = AsyncMock()
        repo.fail_sync = AsyncMock()
        repo.get_by_user_id = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_initial_sync_success(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test successful initial sync."""
        user_id = uuid4()

        # Create async generator for list_all_messages
        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")

        mock_gmail_client.list_all_messages = mock_list

        # Mock batch_get_messages to return raw messages
        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "100",
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Test 1"},
                        {"name": "From", "value": "sender@example.com"},
                        {"name": "Message-ID", "value": "<msg1@example.com>"},
                    ],
                },
            },
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "101",
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Test 2"},
                        {"name": "From", "value": "other@example.com"},
                        {"name": "Message-ID", "value": "<msg2@example.com>"},
                    ],
                },
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.initial_sync(user_id=user_id, days=30)

        assert result.success is True
        assert result.emails_synced == 2
        assert result.history_id == "101"
        assert len(emails) == 2

        mock_sync_repo.start_sync.assert_called_once_with(user_id)
        mock_sync_repo.complete_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_initial_sync_empty(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test initial sync with no messages."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.initial_sync(user_id=user_id, days=30)

        assert result.success is True
        assert result.emails_synced == 0
        assert len(emails) == 0

        mock_sync_repo.complete_sync.assert_called_once_with(user_id, None, 0)

    @pytest.mark.asyncio
    async def test_initial_sync_with_progress_callback(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test initial sync calls progress callback."""
        user_id = uuid4()
        progress_calls: list[tuple[int, int]] = []

        def on_progress(processed: int, total: int) -> None:
            progress_calls.append((processed, total))

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        await service.initial_sync(
            user_id=user_id,
            days=30,
            progress_callback=on_progress,
        )

        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1)

    @pytest.mark.asyncio
    async def test_initial_sync_handles_api_error(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test initial sync handles API errors."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise GmailApiError("API rate limit exceeded", status_code=429)
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.initial_sync(user_id=user_id, days=30)

        assert result.success is False
        assert result.emails_synced == 0
        assert "rate limit" in result.error.lower()

        mock_sync_repo.fail_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_initial_sync_handles_generic_error(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test initial sync handles generic errors."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("Something went wrong")
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.initial_sync(user_id=user_id, days=30)

        assert result.success is False
        assert "Something went wrong" in result.error

        mock_sync_repo.fail_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_initial_sync_skips_failed_messages(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test initial sync skips messages that failed to fetch."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            GmailApiError("Not found", status_code=404),  # Failed message
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.initial_sync(user_id=user_id, days=30)

        assert result.success is True
        assert result.emails_synced == 1
        assert len(emails) == 1

    @pytest.mark.asyncio
    async def test_initial_sync_skips_unparseable_messages(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test initial sync skips messages that can't be parsed."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {},  # Missing id, will fail parse
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.initial_sync(user_id=user_id, days=30)

        assert result.success is True
        assert result.emails_synced == 1


class TestSyncServiceIncrementalSync:
    """Tests for SyncService.incremental_sync method."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        client.get_history = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        repo = MagicMock()
        repo.start_sync = AsyncMock()
        repo.complete_sync = AsyncMock()
        repo.fail_sync = AsyncMock()
        repo.get_by_user_id = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_incremental_sync_no_previous_state(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync falls back to initial sync when no state."""
        user_id = uuid4()
        mock_sync_repo.get_by_user_id.return_value = None

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_incremental_sync_no_history_id(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync falls back when no history ID."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = None
        mock_sync_repo.get_by_user_id.return_value = state

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_incremental_sync_success(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test successful incremental sync."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        # Mock get_history to return new messages
        mock_gmail_client.get_history.return_value = (
            [
                {
                    "messagesAdded": [
                        {"message": {"id": "msg1"}},
                        {"message": {"id": "msg2"}},
                    ]
                }
            ],
            None,  # No next page
            "150",  # New history ID
        )

        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "120",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "150",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True
        assert result.emails_synced == 2
        assert result.history_id == "150"

        mock_sync_repo.complete_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_incremental_sync_no_changes(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync with no new messages."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        mock_gmail_client.get_history.return_value = (
            [],  # No history records
            None,
            "100",
        )

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True
        assert result.emails_synced == 0
        assert len(emails) == 0

    @pytest.mark.asyncio
    async def test_incremental_sync_expired_history(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync falls back when history expired."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        # First call to get_history raises 404
        mock_gmail_client.get_history.side_effect = GmailApiError(
            "History not found", status_code=404
        )

        # Set up for fallback to initial sync
        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        # Should fall back to initial sync
        assert result.success is True

    @pytest.mark.asyncio
    async def test_incremental_sync_api_error(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync handles non-404 API errors."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        mock_gmail_client.get_history.side_effect = GmailApiError("Server error", status_code=500)

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is False
        assert "Server error" in result.error
        mock_sync_repo.fail_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_incremental_sync_with_pagination(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync handles pagination."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        # Mock paginated history
        mock_gmail_client.get_history.side_effect = [
            (
                [{"messagesAdded": [{"message": {"id": "msg1"}}]}],
                "next_token",
                "120",
            ),
            (
                [{"messagesAdded": [{"message": {"id": "msg2"}}]}],
                None,
                "150",
            ),
        ]

        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "120",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "150",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True
        assert result.emails_synced == 2
        assert mock_gmail_client.get_history.call_count == 2


class TestSyncServiceGetSyncStatus:
    """Tests for SyncService.get_sync_status method."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        return MagicMock()

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        repo = MagicMock()
        repo.get_by_user_id = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_get_status_no_state(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test get_sync_status when no state exists."""
        user_id = uuid4()
        mock_sync_repo.get_by_user_id.return_value = None

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        status = await service.get_sync_status(user_id)

        assert status["status"] == "not_synced"
        assert status["emails_synced"] == 0
        assert status["last_sync_at"] is None

    @pytest.mark.asyncio
    async def test_get_status_with_state(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test get_sync_status with existing state."""
        user_id = uuid4()
        now = datetime.now(UTC)

        state = MagicMock()
        state.sync_status = "idle"
        state.emails_synced = 100
        state.last_sync_at = now
        state.last_history_id = "12345"
        state.error_message = None
        mock_sync_repo.get_by_user_id.return_value = state

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        status = await service.get_sync_status(user_id)

        assert status["status"] == "idle"
        assert status["emails_synced"] == 100
        assert status["last_history_id"] == "12345"
        assert status["error"] is None

    @pytest.mark.asyncio
    async def test_get_status_with_error(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test get_sync_status with error state."""
        user_id = uuid4()

        state = MagicMock()
        state.sync_status = "error"
        state.emails_synced = 50
        state.last_sync_at = None
        state.last_history_id = None
        state.error_message = "Connection failed"
        mock_sync_repo.get_by_user_id.return_value = state

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        status = await service.get_sync_status(user_id)

        assert status["status"] == "error"
        assert status["error"] == "Connection failed"


class TestSyncServiceHistoryTracking:
    """Tests for history ID tracking in SyncService."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        repo = MagicMock()
        repo.start_sync = AsyncMock()
        repo.complete_sync = AsyncMock()
        repo.fail_sync = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_initial_sync_tracks_latest_history_id(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test that initial_sync tracks the highest history ID."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")
            yield GmailMessageRef(id="msg3", thread_id="t3")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "200",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "100",  # Lower history ID
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
            {
                "id": "msg3",
                "threadId": "t3",
                "historyId": "300",  # Highest history ID
                "payload": {"headers": [{"name": "Message-ID", "value": "<m3>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, _ = await service.initial_sync(user_id=user_id, days=30)

        assert result.history_id == "300"

    @pytest.mark.asyncio
    async def test_initial_sync_handles_missing_history_id(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test initial_sync handles messages without history_id."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                # No historyId
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.initial_sync(user_id=user_id, days=30)

        assert result.success is True
        assert len(emails) == 2
        assert result.history_id == "100"


class TestSyncServiceIncrementalSyncBranches:
    """Tests for incremental sync branch coverage."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        client.get_history = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        repo = MagicMock()
        repo.start_sync = AsyncMock()
        repo.complete_sync = AsyncMock()
        repo.fail_sync = AsyncMock()
        repo.get_by_user_id = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_incremental_sync_with_progress_callback(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync calls progress callback."""
        user_id = uuid4()
        progress_calls: list[tuple[int, int]] = []

        def on_progress(processed: int, total: int) -> None:
            progress_calls.append((processed, total))

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        mock_gmail_client.get_history.return_value = (
            [{"messagesAdded": [{"message": {"id": "msg1"}}]}],
            None,
            "150",
        )

        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "150",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        await service.incremental_sync(user_id=user_id, progress_callback=on_progress)

        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1)

    @pytest.mark.asyncio
    async def test_incremental_sync_skips_failed_messages(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync skips failed batch messages."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        mock_gmail_client.get_history.return_value = (
            [{"messagesAdded": [{"message": {"id": "msg1"}}, {"message": {"id": "msg2"}}]}],
            None,
            "150",
        )

        mock_gmail_client.batch_get_messages.return_value = [
            GmailApiError("Not found", status_code=404),
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "150",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True
        assert result.emails_synced == 1

    @pytest.mark.asyncio
    async def test_incremental_sync_skips_unparseable_messages(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync skips unparseable messages."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        mock_gmail_client.get_history.return_value = (
            [{"messagesAdded": [{"message": {"id": "msg1"}}, {"message": {"id": "msg2"}}]}],
            None,
            "150",
        )

        mock_gmail_client.batch_get_messages.return_value = [
            {},  # Missing id, will fail parse
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "150",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True
        assert result.emails_synced == 1

    @pytest.mark.asyncio
    async def test_incremental_sync_tracks_latest_history_id(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync tracks latest history ID from messages."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        mock_gmail_client.get_history.return_value = (
            [{"messagesAdded": [{"message": {"id": "msg1"}}, {"message": {"id": "msg2"}}]}],
            None,
            "120",  # History ID from API
        )

        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "150",  # Higher than API history ID
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "140",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, _ = await service.incremental_sync(user_id=user_id)

        assert result.history_id == "150"

    @pytest.mark.asyncio
    async def test_incremental_sync_handles_generic_exception(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync handles generic exceptions."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        mock_gmail_client.get_history.side_effect = RuntimeError("Unexpected error")

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is False
        assert "Unexpected error" in result.error
        mock_sync_repo.fail_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_incremental_sync_handles_malformed_history(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync handles malformed history records."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        # History with various malformed structures
        mock_gmail_client.get_history.return_value = (
            [
                {"messagesAdded": "not a list"},  # messagesAdded not a list
                {"messagesAdded": [123]},  # items not dicts
                {"messagesAdded": [{"message": "not dict"}]},  # message not a dict
                {"messagesAdded": [{"message": {"id": 456}}]},  # id not a string
                {"messagesAdded": [{"message": {"id": "msg1"}}]},  # Valid one
            ],
            None,
            "150",
        )

        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "150",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True
        # Only the valid message should be processed
        assert result.emails_synced == 1

    @pytest.mark.asyncio
    async def test_incremental_sync_with_none_history_id(
        self, mock_gmail_client: MagicMock, mock_sync_repo: MagicMock
    ) -> None:
        """Test incremental sync handles None history_id from API."""
        user_id = uuid4()

        state = MagicMock()
        state.last_history_id = "100"
        mock_sync_repo.get_by_user_id.return_value = state

        # History with None as history_id
        mock_gmail_client.get_history.return_value = (
            [{"messagesAdded": [{"message": {"id": "msg1"}}]}],
            None,  # No next page
            None,  # None history_id - should keep using state.last_history_id
        )

        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "150",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1>"}]},
            },
        ]

        service = SyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
        )

        result, emails = await service.incremental_sync(user_id=user_id)

        assert result.success is True
        assert result.emails_synced == 1
        # Should use history_id from message since API returned None
        assert result.history_id == "150"
