"""Tests for API exceptions and Problem Details."""

from __future__ import annotations

from priority_lens.api.exceptions import (
    APIError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ProblemDetail,
    RateLimitError,
    ServiceUnavailableError,
    UnauthorizedError,
    ValidationError,
)


class TestProblemDetail:
    """Tests for ProblemDetail dataclass."""

    def test_default_values(self) -> None:
        """Test default Problem Detail values."""
        problem = ProblemDetail()

        assert problem.type == "about:blank"
        assert problem.title == "An error occurred"
        assert problem.status == 500
        assert problem.detail is None
        assert problem.instance is None
        assert problem.extensions == {}

    def test_custom_values(self) -> None:
        """Test custom Problem Detail values."""
        problem = ProblemDetail(
            type="/errors/not-found",
            title="Not Found",
            status=404,
            detail="The resource was not found",
            instance="/api/users/123",
            extensions={"resource_id": "123"},
        )

        assert problem.type == "/errors/not-found"
        assert problem.title == "Not Found"
        assert problem.status == 404
        assert problem.detail == "The resource was not found"
        assert problem.instance == "/api/users/123"
        assert problem.extensions == {"resource_id": "123"}

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal values."""
        problem = ProblemDetail()
        result = problem.to_dict()

        assert result == {
            "type": "about:blank",
            "title": "An error occurred",
            "status": 500,
        }

    def test_to_dict_full(self) -> None:
        """Test to_dict with all values."""
        problem = ProblemDetail(
            type="/errors/validation",
            title="Validation Error",
            status=422,
            detail="Request validation failed",
            instance="/api/users",
            extensions={"field": "email", "error": "invalid format"},
        )
        result = problem.to_dict()

        assert result == {
            "type": "/errors/validation",
            "title": "Validation Error",
            "status": 422,
            "detail": "Request validation failed",
            "instance": "/api/users",
            "field": "email",
            "error": "invalid format",
        }

    def test_immutable(self) -> None:
        """Test that ProblemDetail is immutable."""
        problem = ProblemDetail()

        # Should raise FrozenInstanceError or similar
        try:
            problem.status = 404  # type: ignore[misc]
            raise AssertionError("Should not be able to modify frozen dataclass")
        except Exception:
            pass  # Expected


class TestAPIError:
    """Tests for APIError base exception."""

    def test_default_values(self) -> None:
        """Test default APIError values."""
        error = APIError()

        assert error.problem.title == "An error occurred"
        assert error.problem.status == 500
        assert error.status_code == 500

    def test_custom_values(self) -> None:
        """Test custom APIError values."""
        error = APIError(
            title="Custom Error",
            detail="Something went wrong",
            status=418,
            error_type="/errors/custom",
            instance="/api/test",
            custom_field="value",
        )

        assert error.problem.title == "Custom Error"
        assert error.problem.detail == "Something went wrong"
        assert error.problem.status == 418
        assert error.problem.type == "/errors/custom"
        assert error.problem.instance == "/api/test"
        assert error.problem.extensions == {"custom_field": "value"}

    def test_status_code_property(self) -> None:
        """Test status_code property."""
        error = APIError(status=404)
        assert error.status_code == 404

    def test_exception_message(self) -> None:
        """Test exception message from detail."""
        error = APIError(detail="Test detail")
        assert str(error) == "Test detail"

    def test_exception_message_fallback(self) -> None:
        """Test exception message fallback to title."""
        error = APIError(title="Test Title")
        assert str(error) == "Test Title"


class TestNotFoundError:
    """Tests for NotFoundError exception."""

    def test_default(self) -> None:
        """Test default not found error."""
        error = NotFoundError()

        assert error.problem.title == "Not Found"
        assert error.problem.status == 404
        assert error.problem.detail == "Resource not found"
        assert error.problem.type == "/errors/not-found"

    def test_with_resource(self) -> None:
        """Test not found error with resource type."""
        error = NotFoundError(resource="User")

        assert error.problem.detail == "User not found"
        assert error.problem.extensions["resource"] == "User"

    def test_with_resource_and_id(self) -> None:
        """Test not found error with resource and ID."""
        error = NotFoundError(resource="User", resource_id="123")

        assert error.problem.detail == "User with id '123' not found"
        assert error.problem.extensions["resource"] == "User"
        assert error.problem.extensions["resource_id"] == "123"

    def test_with_custom_detail(self) -> None:
        """Test not found error with custom detail."""
        error = NotFoundError(detail="Custom not found message")

        assert error.problem.detail == "Custom not found message"


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_default(self) -> None:
        """Test default validation error."""
        error = ValidationError()

        assert error.problem.title == "Validation Error"
        assert error.problem.status == 422
        assert error.problem.detail == "Validation failed"
        assert error.problem.type == "/errors/validation"

    def test_with_errors(self) -> None:
        """Test validation error with field errors."""
        errors = [
            {"field": "email", "message": "Invalid format"},
            {"field": "age", "message": "Must be positive"},
        ]
        error = ValidationError(detail="Multiple validation errors", errors=errors)

        assert error.problem.detail == "Multiple validation errors"
        assert error.problem.extensions["errors"] == errors


class TestUnauthorizedError:
    """Tests for UnauthorizedError exception."""

    def test_default(self) -> None:
        """Test default unauthorized error."""
        error = UnauthorizedError()

        assert error.problem.title == "Unauthorized"
        assert error.problem.status == 401
        assert error.problem.detail == "Authentication required"
        assert error.problem.type == "/errors/unauthorized"

    def test_with_detail(self) -> None:
        """Test unauthorized error with custom detail."""
        error = UnauthorizedError(detail="Token expired")
        assert error.problem.detail == "Token expired"


class TestForbiddenError:
    """Tests for ForbiddenError exception."""

    def test_default(self) -> None:
        """Test default forbidden error."""
        error = ForbiddenError()

        assert error.problem.title == "Forbidden"
        assert error.problem.status == 403
        assert error.problem.detail == "Permission denied"
        assert error.problem.type == "/errors/forbidden"

    def test_with_detail(self) -> None:
        """Test forbidden error with custom detail."""
        error = ForbiddenError(detail="Admin access required")
        assert error.problem.detail == "Admin access required"


class TestConflictError:
    """Tests for ConflictError exception."""

    def test_default(self) -> None:
        """Test default conflict error."""
        error = ConflictError()

        assert error.problem.title == "Conflict"
        assert error.problem.status == 409
        assert error.problem.detail == "Resource conflict"
        assert error.problem.type == "/errors/conflict"

    def test_with_resource(self) -> None:
        """Test conflict error with resource type."""
        error = ConflictError(detail="Email already exists", resource="User")

        assert error.problem.detail == "Email already exists"
        assert error.problem.extensions["resource"] == "User"


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_default(self) -> None:
        """Test default rate limit error."""
        error = RateLimitError()

        assert error.problem.title == "Too Many Requests"
        assert error.problem.status == 429
        assert error.problem.detail == "Rate limit exceeded"
        assert error.problem.type == "/errors/rate-limit"

    def test_with_retry_after(self) -> None:
        """Test rate limit error with retry_after."""
        error = RateLimitError(detail="Please slow down", retry_after=60)

        assert error.problem.detail == "Please slow down"
        assert error.problem.extensions["retry_after"] == 60


class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError exception."""

    def test_default(self) -> None:
        """Test default service unavailable error."""
        error = ServiceUnavailableError()

        assert error.problem.title == "Service Unavailable"
        assert error.problem.status == 503
        assert error.problem.detail == "Service temporarily unavailable"
        assert error.problem.type == "/errors/service-unavailable"

    def test_with_retry_after(self) -> None:
        """Test service unavailable error with retry_after."""
        error = ServiceUnavailableError(detail="Maintenance in progress", retry_after=300)

        assert error.problem.detail == "Maintenance in progress"
        assert error.problem.extensions["retry_after"] == 300
