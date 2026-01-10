"""Tests for health check routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from priority_lens.api.database import Database
from priority_lens.api.routes.health import HealthResponse, ReadyResponse, router


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_create(self) -> None:
        """Test creating HealthResponse."""
        response = HealthResponse(status="healthy", version="1.0.0")

        assert response.status == "healthy"
        assert response.version == "1.0.0"


class TestReadyResponse:
    """Tests for ReadyResponse model."""

    def test_create(self) -> None:
        """Test creating ReadyResponse."""
        response = ReadyResponse(
            status="ready",
            database="connected",
            checks={"database": True},
        )

        assert response.status == "ready"
        assert response.database == "connected"
        assert response.checks["database"] is True


class TestHealthCheck:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check returns healthy status."""
        from priority_lens.api.routes.health import health_check

        response = await health_check()

        assert response.status == "healthy"
        assert response.version == "2.0.0"


class TestReadinessCheck:
    """Tests for readiness check endpoint."""

    @pytest.mark.asyncio
    async def test_readiness_check_healthy(self) -> None:
        """Test readiness check when database is healthy."""
        from priority_lens.api.routes.health import readiness_check

        mock_db = AsyncMock(spec=Database)
        mock_db.check_connection.return_value = True

        response = await readiness_check(mock_db)

        assert response.status == "ready"
        assert response.database == "connected"
        assert response.checks["database"] is True

    @pytest.mark.asyncio
    async def test_readiness_check_unhealthy(self) -> None:
        """Test readiness check when database is unhealthy."""
        from priority_lens.api.routes.health import readiness_check

        mock_db = AsyncMock(spec=Database)
        mock_db.check_connection.return_value = False

        with patch("priority_lens.api.routes.health.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            response = await readiness_check(mock_db)

        assert response.status == "not_ready"
        assert response.database == "disconnected"
        assert response.checks["database"] is False

    @pytest.mark.asyncio
    async def test_readiness_check_logs_failure(self) -> None:
        """Test that readiness failure is logged."""
        from priority_lens.api.routes.health import readiness_check

        mock_db = AsyncMock(spec=Database)
        mock_db.check_connection.return_value = False

        with patch("priority_lens.api.routes.health.logger") as mock_logger:
            mock_logger.awarning = AsyncMock()
            await readiness_check(mock_db)

            mock_logger.awarning.assert_called_once()


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_prefix(self) -> None:
        """Test that router has correct prefix."""
        assert router.prefix == "/health"

    def test_router_tags(self) -> None:
        """Test that router has correct tags."""
        assert "health" in router.tags
