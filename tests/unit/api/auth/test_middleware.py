"""Tests for authentication middleware."""

from __future__ import annotations

from unittest import mock

import pytest
from starlette.applications import Starlette
from starlette.datastructures import State
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient

from priority_lens.api.auth.clerk import ClerkUser
from priority_lens.api.auth.config import ClerkConfig
from priority_lens.api.auth.middleware import (
    AuthenticationMiddleware,
    add_authentication_middleware,
)


class TestAuthenticationMiddleware:
    """Tests for AuthenticationMiddleware."""

    def test_initializes_user_state_to_none(self) -> None:
        """Test middleware initializes user state to None."""
        captured_state: list[State] = []

        async def capture_state(request: Request) -> Response:
            captured_state.append(request.state)
            return Response("OK")

        app = Starlette(routes=[Route("/", capture_state)])
        middleware = AuthenticationMiddleware(app)
        client = TestClient(middleware)

        client.get("/")

        assert len(captured_state) == 1
        assert captured_state[0].user is None

    def test_passes_through_without_auth_header(self) -> None:
        """Test middleware passes through when no auth header."""

        async def endpoint(request: Request) -> Response:
            return Response("OK", status_code=200)

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)
        client = TestClient(middleware)

        response = client.get("/")

        assert response.status_code == 200
        assert response.text == "OK"

    def test_attempts_authentication_with_bearer_token(self) -> None:
        """Test middleware attempts authentication with bearer token."""
        user_from_request: list[ClerkUser | None] = []

        async def endpoint(request: Request) -> Response:
            user_from_request.append(getattr(request.state, "user", None))
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)

        # Mock the validator by setting _validator directly
        mock_user = ClerkUser(id="user_123", email="user@example.com")
        mock_validator = mock.MagicMock()
        mock_validator.validate_token.return_value = mock_user
        middleware._validator = mock_validator

        client = TestClient(middleware)
        client.get("/", headers={"Authorization": "Bearer valid-token"})

        assert len(user_from_request) == 1
        assert user_from_request[0] is not None
        assert user_from_request[0].id == "user_123"

    def test_ignores_non_bearer_schemes(self) -> None:
        """Test middleware ignores non-bearer auth schemes."""
        user_from_request: list[ClerkUser | None] = []

        async def endpoint(request: Request) -> Response:
            user_from_request.append(getattr(request.state, "user", None))
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)
        client = TestClient(middleware)

        client.get("/", headers={"Authorization": "Basic dXNlcjpwYXNz"})

        assert len(user_from_request) == 1
        assert user_from_request[0] is None

    def test_continues_on_validation_failure(self) -> None:
        """Test middleware continues when validation fails."""

        async def endpoint(request: Request) -> Response:
            user = getattr(request.state, "user", "NOT_SET")
            return Response(f"user: {user}")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)

        # Mock the validator by setting _validator directly
        mock_validator = mock.MagicMock()
        mock_validator.validate_token.side_effect = Exception("Invalid token")
        middleware._validator = mock_validator

        client = TestClient(middleware)
        response = client.get("/", headers={"Authorization": "Bearer bad-token"})

        # Should still get 200, user should be None
        assert response.status_code == 200
        assert "None" in response.text

    def test_lazy_validator_initialization(self) -> None:
        """Test validator is lazily initialized."""

        async def endpoint(request: Request) -> Response:
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)

        # Initially no validator
        assert middleware._validator is None

    def test_validator_not_created_when_unconfigured(self) -> None:
        """Test validator returns None when Clerk not configured."""

        async def endpoint(request: Request) -> Response:
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)

        # Mock get_clerk_config to return unconfigured config
        with mock.patch("priority_lens.api.auth.middleware.get_clerk_config") as mock_get_config:
            mock_get_config.return_value = ClerkConfig(secret_key="", issuer="")  # Not configured

            # Access validator property
            validator = middleware.validator

            assert validator is None

    def test_validator_created_when_configured(self) -> None:
        """Test validator is created when Clerk is configured."""

        async def endpoint(request: Request) -> Response:
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)

        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )

        with mock.patch("priority_lens.api.auth.middleware.get_clerk_config") as mock_get_config:
            mock_get_config.return_value = config

            validator = middleware.validator

            assert validator is not None

    def test_skips_auth_when_validator_is_none(self) -> None:
        """Test middleware skips auth attempt when validator is None.

        This tests the dispatch path where validator is None, so
        _try_authenticate is never called.
        """
        captured_user: list[ClerkUser | None] = []

        async def endpoint(request: Request) -> Response:
            captured_user.append(getattr(request.state, "user", None))
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)

        # Mock get_clerk_config to return unconfigured config
        # This ensures validator property returns None
        with mock.patch("priority_lens.api.auth.middleware.get_clerk_config") as mock_get_config:
            mock_get_config.return_value = ClerkConfig()  # Not configured

            client = TestClient(middleware)
            response = client.get("/", headers={"Authorization": "Bearer some-token"})

        assert response.status_code == 200
        assert len(captured_user) == 1
        # User should be None because auth was skipped (validator was None)
        assert captured_user[0] is None

    def test_handles_malformed_bearer_token(self) -> None:
        """Test middleware handles malformed bearer token format."""
        captured_user: list[ClerkUser | None] = []

        async def endpoint(request: Request) -> Response:
            captured_user.append(getattr(request.state, "user", None))
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        middleware = AuthenticationMiddleware(app)

        # Set a valid validator to ensure _try_authenticate is called
        mock_validator = mock.MagicMock()
        middleware._validator = mock_validator

        client = TestClient(middleware)
        # Send token with "Bearer" but no actual token after space
        response = client.get("/", headers={"Authorization": "Bearer"})

        assert response.status_code == 200
        assert len(captured_user) == 1
        assert captured_user[0] is None
        # validate_token should not have been called since format was invalid
        mock_validator.validate_token.assert_not_called()


class TestAddAuthenticationMiddleware:
    """Tests for add_authentication_middleware factory."""

    def test_wraps_app_with_middleware(self) -> None:
        """Test factory wraps app with AuthenticationMiddleware."""

        async def endpoint(request: Request) -> Response:
            return Response("OK")

        app = Starlette(routes=[Route("/", endpoint)])
        wrapped = add_authentication_middleware(app)

        assert isinstance(wrapped, AuthenticationMiddleware)

    def test_wrapped_app_works(self) -> None:
        """Test wrapped app functions correctly."""

        async def endpoint(request: Request) -> Response:
            return Response("Hello", status_code=200)

        app = Starlette(routes=[Route("/", endpoint)])
        wrapped = add_authentication_middleware(app)
        client = TestClient(wrapped)

        response = client.get("/")

        assert response.status_code == 200
        assert response.text == "Hello"


class TestMiddlewareIntegration:
    """Integration tests for authentication middleware."""

    @pytest.fixture
    def configured_config(self) -> ClerkConfig:
        """Create configured Clerk config."""
        return ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )

    def test_full_auth_flow(self, configured_config: ClerkConfig) -> None:
        """Test full authentication flow through middleware."""
        captured_user: list[ClerkUser | None] = []

        async def protected_endpoint(request: Request) -> Response:
            captured_user.append(getattr(request.state, "user", None))
            if request.state.user:
                return Response(f"Hello {request.state.user.id}")
            return Response("Anonymous", status_code=200)

        app = Starlette(routes=[Route("/", protected_endpoint)])
        middleware = AuthenticationMiddleware(app)

        expected_user = ClerkUser(id="user_abc", email="test@example.com")

        with mock.patch.object(middleware, "_validator") as mock_validator:
            mock_validator.validate_token.return_value = expected_user

            # Set _validator directly so property returns it
            middleware._validator = mock_validator

            client = TestClient(middleware)
            response = client.get("/", headers={"Authorization": "Bearer test-token"})

        assert response.status_code == 200
        assert "user_abc" in response.text
        assert len(captured_user) == 1
        assert captured_user[0] is not None
        assert captured_user[0].id == "user_abc"

    def test_anonymous_access(self) -> None:
        """Test anonymous access without token."""
        captured_user: list[ClerkUser | None] = []

        async def public_endpoint(request: Request) -> Response:
            captured_user.append(getattr(request.state, "user", None))
            return Response("Public content")

        app = Starlette(routes=[Route("/", public_endpoint)])
        middleware = AuthenticationMiddleware(app)
        client = TestClient(middleware)

        response = client.get("/")

        assert response.status_code == 200
        assert response.text == "Public content"
        assert len(captured_user) == 1
        assert captured_user[0] is None
