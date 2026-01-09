"""Custom exceptions and RFC 7807 Problem Details support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProblemDetail:
    """RFC 7807 Problem Details response.

    See: https://datatracker.ietf.org/doc/html/rfc7807
    """

    type: str = field(default="about:blank")
    title: str = field(default="An error occurred")
    status: int = field(default=500)
    detail: str | None = field(default=None)
    instance: str | None = field(default=None)
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response.

        Returns:
            Dictionary representation of the problem detail.
        """
        result: dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "status": self.status,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        if self.instance is not None:
            result["instance"] = self.instance
        result.update(self.extensions)
        return result


class APIError(Exception):
    """Base exception for API errors with Problem Details support."""

    def __init__(
        self,
        title: str = "An error occurred",
        detail: str | None = None,
        status: int = 500,
        error_type: str = "about:blank",
        instance: str | None = None,
        **extensions: Any,
    ) -> None:
        """Initialize API error.

        Args:
            title: Short human-readable summary.
            detail: Detailed explanation of the error.
            status: HTTP status code.
            error_type: URI reference identifying the problem type.
            instance: URI reference identifying the specific occurrence.
            **extensions: Additional fields to include in the response.
        """
        super().__init__(detail or title)
        self.problem = ProblemDetail(
            type=error_type,
            title=title,
            status=status,
            detail=detail,
            instance=instance,
            extensions=extensions,
        )

    @property
    def status_code(self) -> int:
        """Get the HTTP status code."""
        return self.problem.status


class NotFoundError(APIError):
    """Resource not found error (404)."""

    def __init__(
        self,
        resource: str = "Resource",
        resource_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Initialize not found error.

        Args:
            resource: Type of resource that was not found.
            resource_id: Identifier of the resource.
            detail: Additional details about the error.
        """
        message = detail or f"{resource} not found"
        if resource_id and not detail:
            message = f"{resource} with id '{resource_id}' not found"

        super().__init__(
            title="Not Found",
            detail=message,
            status=404,
            error_type="/errors/not-found",
            resource=resource,
            resource_id=resource_id,
        )


class ValidationError(APIError):
    """Request validation error (422)."""

    def __init__(
        self,
        detail: str = "Validation failed",
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            detail: Description of what validation failed.
            errors: List of specific validation errors.
        """
        super().__init__(
            title="Validation Error",
            detail=detail,
            status=422,
            error_type="/errors/validation",
            errors=errors or [],
        )


class UnauthorizedError(APIError):
    """Authentication required error (401)."""

    def __init__(self, detail: str = "Authentication required") -> None:
        """Initialize unauthorized error.

        Args:
            detail: Description of the authentication requirement.
        """
        super().__init__(
            title="Unauthorized",
            detail=detail,
            status=401,
            error_type="/errors/unauthorized",
        )


class ForbiddenError(APIError):
    """Permission denied error (403)."""

    def __init__(self, detail: str = "Permission denied") -> None:
        """Initialize forbidden error.

        Args:
            detail: Description of why permission was denied.
        """
        super().__init__(
            title="Forbidden",
            detail=detail,
            status=403,
            error_type="/errors/forbidden",
        )


class ConflictError(APIError):
    """Resource conflict error (409)."""

    def __init__(
        self,
        detail: str = "Resource conflict",
        resource: str | None = None,
    ) -> None:
        """Initialize conflict error.

        Args:
            detail: Description of the conflict.
            resource: Type of resource with the conflict.
        """
        super().__init__(
            title="Conflict",
            detail=detail,
            status=409,
            error_type="/errors/conflict",
            resource=resource,
        )


class RateLimitError(APIError):
    """Rate limit exceeded error (429)."""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            detail: Description of the rate limit.
            retry_after: Seconds until the client can retry.
        """
        extensions: dict[str, Any] = {}
        if retry_after is not None:
            extensions["retry_after"] = retry_after

        super().__init__(
            title="Too Many Requests",
            detail=detail,
            status=429,
            error_type="/errors/rate-limit",
            **extensions,
        )


class ServiceUnavailableError(APIError):
    """Service temporarily unavailable error (503)."""

    def __init__(
        self,
        detail: str = "Service temporarily unavailable",
        retry_after: int | None = None,
    ) -> None:
        """Initialize service unavailable error.

        Args:
            detail: Description of the service issue.
            retry_after: Seconds until the service may be available.
        """
        extensions: dict[str, Any] = {}
        if retry_after is not None:
            extensions["retry_after"] = retry_after

        super().__init__(
            title="Service Unavailable",
            detail=detail,
            status=503,
            error_type="/errors/service-unavailable",
            **extensions,
        )
