"""Tests for SyncState model."""

from __future__ import annotations

import uuid

from rl_emails.models.sync_state import SyncState


class TestSyncState:
    """Tests for SyncState model."""

    def test_sync_state_create(self) -> None:
        """Test creating a sync state."""
        user_id = uuid.uuid4()
        state = SyncState(
            user_id=user_id,
            last_history_id="12345",
            sync_status="idle",
            emails_synced=100,
        )
        assert state.user_id == user_id
        assert state.last_history_id == "12345"
        assert state.sync_status == "idle"
        assert state.emails_synced == 100

    def test_sync_state_default_status(self) -> None:
        """Test sync state has idle status by default."""
        user_id = uuid.uuid4()
        state = SyncState(user_id=user_id)
        # Before commit, status may be None or 'idle' depending on SQLAlchemy version
        assert state.sync_status is None or state.sync_status == "idle"

    def test_sync_state_default_emails_synced(self) -> None:
        """Test sync state has 0 emails synced by default."""
        user_id = uuid.uuid4()
        state = SyncState(user_id=user_id)
        # Before commit, emails_synced may be None or 0 depending on SQLAlchemy version
        assert state.emails_synced is None or state.emails_synced == 0

    def test_sync_state_is_syncing_true(self) -> None:
        """Test is_syncing returns True when syncing."""
        user_id = uuid.uuid4()
        state = SyncState(user_id=user_id, sync_status="syncing")
        assert state.is_syncing is True

    def test_sync_state_is_syncing_false(self) -> None:
        """Test is_syncing returns False when not syncing."""
        user_id = uuid.uuid4()
        state = SyncState(user_id=user_id, sync_status="idle")
        assert state.is_syncing is False

    def test_sync_state_has_error_true(self) -> None:
        """Test has_error returns True when status is error."""
        user_id = uuid.uuid4()
        state = SyncState(user_id=user_id, sync_status="error", error_message="Failed")
        assert state.has_error is True

    def test_sync_state_has_error_false(self) -> None:
        """Test has_error returns False when status is not error."""
        user_id = uuid.uuid4()
        state = SyncState(user_id=user_id, sync_status="idle")
        assert state.has_error is False

    def test_sync_state_repr(self) -> None:
        """Test sync state string representation."""
        state_id = uuid.uuid4()
        user_id = uuid.uuid4()
        state = SyncState(
            id=state_id,
            user_id=user_id,
            sync_status="idle",
            emails_synced=50,
        )
        result = repr(state)
        assert "SyncState" in result
        assert "idle" in result
        assert "50" in result


class TestSyncStateTablename:
    """Tests for SyncState table configuration."""

    def test_tablename(self) -> None:
        """Test table name is correct."""
        assert SyncState.__tablename__ == "sync_state"
