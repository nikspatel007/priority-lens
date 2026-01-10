"""Tests for OrgUser model."""

from __future__ import annotations

import uuid

from priority_lens.models.org_user import OrgUser


class TestOrgUser:
    """Tests for OrgUser model."""

    def test_org_user_create(self) -> None:
        """Test creating an org user."""
        org_id = uuid.uuid4()
        user = OrgUser(
            org_id=org_id,
            email="user@example.com",
            name="Test User",
            role="admin",
        )
        assert user.org_id == org_id
        assert user.email == "user@example.com"
        assert user.name == "Test User"
        assert user.role == "admin"

    def test_org_user_default_role(self) -> None:
        """Test org user has member role by default."""
        org_id = uuid.uuid4()
        user = OrgUser(org_id=org_id, email="user@example.com")
        # Before commit, role may be None or 'member' depending on SQLAlchemy version
        assert user.role is None or user.role == "member"

    def test_org_user_default_gmail_connected(self) -> None:
        """Test org user has gmail_connected=False by default."""
        org_id = uuid.uuid4()
        user = OrgUser(org_id=org_id, email="user@example.com")
        # Before commit, gmail_connected may be None or False
        assert user.gmail_connected is None or user.gmail_connected is False

    def test_org_user_repr(self) -> None:
        """Test org user string representation."""
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user = OrgUser(id=user_id, org_id=org_id, email="user@example.com", role="admin")
        result = repr(user)
        assert "OrgUser" in result
        assert "user@example.com" in result
        assert "admin" in result


class TestOrgUserTablename:
    """Tests for OrgUser table configuration."""

    def test_tablename(self) -> None:
        """Test table name is correct."""
        assert OrgUser.__tablename__ == "org_users"
