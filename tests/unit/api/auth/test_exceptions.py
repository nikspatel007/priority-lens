"""Tests for authentication exceptions."""

from __future__ import annotations

from priority_lens.api.auth.exceptions import (
    AuthenticationError,
    InsufficientPermissionsError,
    InvalidTokenError,
    TokenExpiredError,
)


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = AuthenticationError()

        assert error.problem.title == "Unauthorized"
        assert error.problem.status == 401
        assert error.problem.detail == "Authentication failed"
        assert error.problem.type == "/errors/authentication"

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = AuthenticationError(detail="Custom auth failure")

        assert error.problem.detail == "Custom auth failure"

    def test_status_code(self) -> None:
        """Test status code property."""
        error = AuthenticationError()

        assert error.status_code == 401


class TestInvalidTokenError:
    """Tests for InvalidTokenError exception."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = InvalidTokenError()

        assert error.problem.title == "Invalid Token"
        assert error.problem.status == 401
        assert error.problem.detail == "Invalid or malformed token"
        assert error.problem.type == "/errors/invalid-token"

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = InvalidTokenError(detail="Missing required claim")

        assert error.problem.detail == "Missing required claim"

    def test_inherits_from_authentication_error(self) -> None:
        """Test InvalidTokenError inherits from AuthenticationError."""
        error = InvalidTokenError()

        assert isinstance(error, AuthenticationError)


class TestTokenExpiredError:
    """Tests for TokenExpiredError exception."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = TokenExpiredError()

        assert error.problem.title == "Token Expired"
        assert error.problem.status == 401
        assert error.problem.detail == "Token has expired"
        assert error.problem.type == "/errors/token-expired"

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = TokenExpiredError(detail="Token expired 5 minutes ago")

        assert error.problem.detail == "Token expired 5 minutes ago"

    def test_inherits_from_authentication_error(self) -> None:
        """Test TokenExpiredError inherits from AuthenticationError."""
        error = TokenExpiredError()

        assert isinstance(error, AuthenticationError)


class TestInsufficientPermissionsError:
    """Tests for InsufficientPermissionsError exception."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = InsufficientPermissionsError()

        assert error.problem.title == "Forbidden"
        assert error.problem.status == 403
        assert error.problem.detail == "Insufficient permissions"
        assert error.problem.type == "/errors/insufficient-permissions"

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = InsufficientPermissionsError(detail="Admin role required")

        assert error.problem.detail == "Admin role required"

    def test_status_code(self) -> None:
        """Test status code property."""
        error = InsufficientPermissionsError()

        assert error.status_code == 403
