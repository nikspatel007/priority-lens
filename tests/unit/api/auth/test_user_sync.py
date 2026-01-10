"""Tests for user synchronization service."""

from __future__ import annotations

import uuid
from unittest import mock

import pytest

from priority_lens.api.auth.clerk import ClerkUser
from priority_lens.api.auth.user_sync import UserSyncService
from priority_lens.models.org_user import OrgUser


class TestUserSyncService:
    """Tests for UserSyncService."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> UserSyncService:
        """Create UserSyncService with mocked session."""
        return UserSyncService(mock_session)

    @pytest.fixture
    def org_id(self) -> uuid.UUID:
        """Create test organization ID."""
        return uuid.uuid4()

    @pytest.fixture
    def clerk_user(self) -> ClerkUser:
        """Create test Clerk user."""
        return ClerkUser(
            id="clerk_user_123",
            email="user@example.com",
            first_name="John",
            last_name="Doe",
        )

    @pytest.fixture
    def existing_org_user(self, org_id: uuid.UUID) -> OrgUser:
        """Create existing OrgUser."""
        user = mock.MagicMock(spec=OrgUser)
        user.id = uuid.uuid4()
        user.org_id = org_id
        user.email = "user@example.com"
        user.name = "John Doe"
        user.role = "member"
        user.gmail_connected = False
        return user


class TestSyncUser:
    """Tests for sync_user method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> UserSyncService:
        """Create UserSyncService with mocked session."""
        return UserSyncService(mock_session)

    @pytest.fixture
    def org_id(self) -> uuid.UUID:
        """Create test organization ID."""
        return uuid.uuid4()

    @pytest.mark.anyio
    async def test_raises_without_email(self, service: UserSyncService, org_id: uuid.UUID) -> None:
        """Test raises ValueError when Clerk user has no email."""
        clerk_user = ClerkUser(id="user_123")  # No email

        with pytest.raises(ValueError, match="must have an email"):
            await service.sync_user(clerk_user, org_id)

    @pytest.mark.anyio
    async def test_creates_new_user_when_not_found(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test creates new user when not found in database."""
        clerk_user = ClerkUser(
            id="clerk_123",
            email="new@example.com",
            first_name="New",
            last_name="User",
        )

        new_org_user = mock.MagicMock(spec=OrgUser)
        new_org_user.id = uuid.uuid4()
        new_org_user.email = "new@example.com"
        new_org_user.name = "New User"

        with mock.patch.object(service._repo, "get_by_email") as mock_get:
            mock_get.return_value = None

            with mock.patch.object(service._repo, "create") as mock_create:
                mock_create.return_value = new_org_user

                result = await service.sync_user(clerk_user, org_id)

                mock_get.assert_called_once_with(org_id, "new@example.com")
                mock_create.assert_called_once()
                assert result == new_org_user

    @pytest.mark.anyio
    async def test_returns_existing_user_when_found(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test returns existing user when found by email."""
        clerk_user = ClerkUser(
            id="clerk_123",
            email="existing@example.com",
            first_name="Existing",
            last_name="User",
        )

        existing_user = mock.MagicMock(spec=OrgUser)
        existing_user.id = uuid.uuid4()
        existing_user.email = "existing@example.com"
        existing_user.name = "Existing User"

        with mock.patch.object(service._repo, "get_by_email") as mock_get:
            mock_get.return_value = existing_user

            with mock.patch.object(service._repo, "update") as mock_update:
                result = await service.sync_user(clerk_user, org_id)

                mock_get.assert_called_once_with(org_id, "existing@example.com")
                # Should not update since name matches
                mock_update.assert_not_called()
                assert result == existing_user

    @pytest.mark.anyio
    async def test_updates_name_when_changed(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test updates user name when it has changed."""
        clerk_user = ClerkUser(
            id="clerk_123",
            email="user@example.com",
            first_name="New",
            last_name="Name",
        )

        existing_user = mock.MagicMock(spec=OrgUser)
        existing_user.id = uuid.uuid4()
        existing_user.email = "user@example.com"
        existing_user.name = "Old Name"

        with mock.patch.object(service._repo, "get_by_email") as mock_get:
            mock_get.return_value = existing_user

            with mock.patch.object(service._repo, "update") as mock_update:
                await service.sync_user(clerk_user, org_id)

                mock_update.assert_called_once()
                call_args = mock_update.call_args
                assert call_args[0][0] == existing_user.id
                assert call_args[0][1].name == "New Name"

    @pytest.mark.anyio
    async def test_does_not_update_when_name_matches(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test does not update when name already matches."""
        clerk_user = ClerkUser(
            id="clerk_123",
            email="user@example.com",
            first_name="Same",
            last_name="Name",
        )

        existing_user = mock.MagicMock(spec=OrgUser)
        existing_user.id = uuid.uuid4()
        existing_user.email = "user@example.com"
        existing_user.name = "Same Name"

        with mock.patch.object(service._repo, "get_by_email") as mock_get:
            mock_get.return_value = existing_user

            with mock.patch.object(service._repo, "update") as mock_update:
                await service.sync_user(clerk_user, org_id)

                mock_update.assert_not_called()


class TestGetByClerkEmail:
    """Tests for get_by_clerk_email method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> UserSyncService:
        """Create UserSyncService with mocked session."""
        return UserSyncService(mock_session)

    @pytest.fixture
    def org_id(self) -> uuid.UUID:
        """Create test organization ID."""
        return uuid.uuid4()

    @pytest.mark.anyio
    async def test_returns_user_when_found(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test returns user when found by email."""
        expected_user = mock.MagicMock(spec=OrgUser)
        expected_user.email = "found@example.com"

        with mock.patch.object(service._repo, "get_by_email") as mock_get:
            mock_get.return_value = expected_user

            result = await service.get_by_clerk_email("found@example.com", org_id)

            mock_get.assert_called_once_with(org_id, "found@example.com")
            assert result == expected_user

    @pytest.mark.anyio
    async def test_returns_none_when_not_found(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test returns None when user not found."""
        with mock.patch.object(service._repo, "get_by_email") as mock_get:
            mock_get.return_value = None

            result = await service.get_by_clerk_email("notfound@example.com", org_id)

            assert result is None


class TestEnsureUserExists:
    """Tests for ensure_user_exists method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> UserSyncService:
        """Create UserSyncService with mocked session."""
        return UserSyncService(mock_session)

    @pytest.fixture
    def org_id(self) -> uuid.UUID:
        """Create test organization ID."""
        return uuid.uuid4()

    @pytest.mark.anyio
    async def test_delegates_to_sync_user(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test ensure_user_exists delegates to sync_user."""
        clerk_user = ClerkUser(id="clerk_123", email="test@example.com")
        expected_user = mock.MagicMock(spec=OrgUser)

        with mock.patch.object(service, "sync_user") as mock_sync:
            mock_sync.return_value = expected_user

            result = await service.ensure_user_exists(clerk_user, org_id)

            mock_sync.assert_called_once_with(clerk_user, org_id)
            assert result == expected_user


class TestCreateUser:
    """Tests for _create_user private method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> UserSyncService:
        """Create UserSyncService with mocked session."""
        return UserSyncService(mock_session)

    @pytest.fixture
    def org_id(self) -> uuid.UUID:
        """Create test organization ID."""
        return uuid.uuid4()

    @pytest.mark.anyio
    async def test_creates_user_with_correct_data(
        self, service: UserSyncService, org_id: uuid.UUID
    ) -> None:
        """Test _create_user creates user with correct data."""
        clerk_user = ClerkUser(
            id="clerk_123",
            email="new@example.com",
            first_name="First",
            last_name="Last",
        )

        created_user = mock.MagicMock(spec=OrgUser)
        created_user.email = "new@example.com"
        created_user.name = "First Last"
        created_user.role = "member"

        with mock.patch.object(service._repo, "create") as mock_create:
            mock_create.return_value = created_user

            result = await service._create_user(clerk_user, org_id)

            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[0][0] == org_id
            create_data = call_args[0][1]
            assert create_data.email == "new@example.com"
            assert create_data.name == "First Last"
            assert create_data.role == "member"
            assert result == created_user

    @pytest.mark.anyio
    async def test_raises_without_email(self, service: UserSyncService, org_id: uuid.UUID) -> None:
        """Test _create_user raises when no email."""
        clerk_user = ClerkUser(id="clerk_123")  # No email

        with pytest.raises(ValueError, match="must have an email"):
            await service._create_user(clerk_user, org_id)
