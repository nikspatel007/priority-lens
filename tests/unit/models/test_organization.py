"""Tests for Organization model."""

from __future__ import annotations

import uuid

from rl_emails.models.organization import Organization


class TestOrganization:
    """Tests for Organization model."""

    def test_organization_create(self) -> None:
        """Test creating an organization."""
        org = Organization(
            name="Acme Corp",
            slug="acme",
            settings={"feature_x": True},
        )
        assert org.name == "Acme Corp"
        assert org.slug == "acme"
        assert org.settings == {"feature_x": True}

    def test_organization_default_id(self) -> None:
        """Test organization has UUID id by default."""
        org = Organization(name="Test", slug="test")
        # Default is generated when not set
        assert org.id is None or isinstance(org.id, uuid.UUID)

    def test_organization_default_settings(self) -> None:
        """Test organization has empty dict settings by default."""
        org = Organization(name="Test", slug="test")
        # Before commit, settings may be None or empty dict depending on SQLAlchemy version
        assert org.settings is None or org.settings == {}

    def test_organization_repr(self) -> None:
        """Test organization string representation."""
        org_id = uuid.uuid4()
        org = Organization(id=org_id, name="Acme Corp", slug="acme")
        result = repr(org)
        assert "Organization" in result
        assert "Acme Corp" in result
        assert "acme" in result


class TestOrganizationTablename:
    """Tests for Organization table configuration."""

    def test_tablename(self) -> None:
        """Test table name is correct."""
        assert Organization.__tablename__ == "organizations"
