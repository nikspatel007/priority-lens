"""Tests for Clerk JWT validation and user extraction."""

from __future__ import annotations

import time
from typing import Any
from unittest import mock

import jwt
import pytest

from rl_emails.api.auth.clerk import ClerkJWTValidator, ClerkUser
from rl_emails.api.auth.config import ClerkConfig
from rl_emails.api.auth.exceptions import InvalidTokenError, TokenExpiredError


class TestClerkUser:
    """Tests for ClerkUser dataclass."""

    def test_minimal_user(self) -> None:
        """Test creating user with only required field."""
        user = ClerkUser(id="user_123")

        assert user.id == "user_123"
        assert user.email is None
        assert user.first_name is None
        assert user.last_name is None
        assert user.image_url is None
        assert user.metadata == {}

    def test_full_user(self) -> None:
        """Test creating user with all fields."""
        metadata = {"public": {"role": "admin"}}
        user = ClerkUser(
            id="user_123",
            email="user@example.com",
            first_name="John",
            last_name="Doe",
            image_url="https://example.com/avatar.jpg",
            metadata=metadata,
        )

        assert user.id == "user_123"
        assert user.email == "user@example.com"
        assert user.first_name == "John"
        assert user.last_name == "Doe"
        assert user.image_url == "https://example.com/avatar.jpg"
        assert user.metadata == metadata

    def test_full_name_both_names(self) -> None:
        """Test full_name with both first and last name."""
        user = ClerkUser(id="user_123", first_name="John", last_name="Doe")

        assert user.full_name == "John Doe"

    def test_full_name_first_only(self) -> None:
        """Test full_name with only first name."""
        user = ClerkUser(id="user_123", first_name="John")

        assert user.full_name == "John"

    def test_full_name_last_only(self) -> None:
        """Test full_name with only last name."""
        user = ClerkUser(id="user_123", last_name="Doe")

        assert user.full_name == "Doe"

    def test_full_name_none(self) -> None:
        """Test full_name with no names."""
        user = ClerkUser(id="user_123")

        assert user.full_name is None

    def test_immutable(self) -> None:
        """Test that ClerkUser is immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        user = ClerkUser(id="user_123")

        with pytest.raises(FrozenInstanceError):
            user.id = "different"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Test that ClerkUser supports equality."""
        user1 = ClerkUser(id="user_123", email="test@example.com")
        user2 = ClerkUser(id="user_123", email="test@example.com")
        user3 = ClerkUser(id="user_456", email="other@example.com")

        assert user1 == user2
        assert user1 != user3


class TestClerkJWTValidator:
    """Tests for ClerkJWTValidator."""

    @pytest.fixture
    def config(self) -> ClerkConfig:
        """Create test config."""
        return ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            audience="",
            token_leeway_seconds=30,
        )

    @pytest.fixture
    def config_with_audience(self) -> ClerkConfig:
        """Create test config with audience."""
        return ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
            audience="my-app",
        )

    def test_init_requires_jwks_url(self) -> None:
        """Test validator requires JWKS URL."""
        config = ClerkConfig()  # No issuer or jwks_url

        with pytest.raises(ValueError, match="JWKS URL is required"):
            ClerkJWTValidator(config)

    def test_init_with_derived_jwks_url(self) -> None:
        """Test validator works with JWKS URL derived from issuer."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )

        validator = ClerkJWTValidator(config)

        assert validator._config == config

    def test_jwks_client_lazy_init(self, config: ClerkConfig) -> None:
        """Test JWKS client is lazily initialized."""
        validator = ClerkJWTValidator(config)

        assert validator._jwks_client is None

        # Access property triggers initialization
        _ = validator.jwks_client

        assert validator._jwks_client is not None

    def test_validate_token_invalid_format(self, config: ClerkConfig) -> None:
        """Test validation fails for malformed token."""
        validator = ClerkJWTValidator(config)

        with pytest.raises(InvalidTokenError):
            validator.validate_token("not-a-valid-jwt")

    def test_validate_token_expired(self, config: ClerkConfig) -> None:
        """Test validation fails for expired token."""
        validator = ClerkJWTValidator(config)

        # Mock the JWKS client to avoid network call
        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        # Mock jwt.decode to raise ExpiredSignatureError
        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

            with pytest.raises(TokenExpiredError):
                validator.validate_token("expired.token.here")

    def test_validate_token_invalid_signature(self, config: ClerkConfig) -> None:
        """Test validation fails for invalid signature."""
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.side_effect = jwt.InvalidSignatureError("Invalid signature")

            with pytest.raises(InvalidTokenError):
                validator.validate_token("invalid.signature.token")

    def test_validate_token_generic_error(self, config: ClerkConfig) -> None:
        """Test validation handles generic errors."""
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.side_effect = Exception("Something went wrong")

            with pytest.raises(InvalidTokenError, match="Token validation failed"):
                validator.validate_token("problematic.token.here")

    def test_validate_token_success(self, config: ClerkConfig) -> None:
        """Test successful token validation."""
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        payload = {
            "sub": "user_123",
            "email": "user@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            user = validator.validate_token("valid.token.here")

            assert user.id == "user_123"
            assert user.email == "user@example.com"
            assert user.first_name == "John"
            assert user.last_name == "Doe"

    def test_validate_token_with_nested_user_data(self, config: ClerkConfig) -> None:
        """Test validation extracts user data from nested location."""
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        payload = {
            "sub": "user_456",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "user": {
                "primary_email_address": "nested@example.com",
                "first_name": "Jane",
                "last_name": "Smith",
                "image_url": "https://example.com/jane.jpg",
            },
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            user = validator.validate_token("valid.token.here")

            assert user.id == "user_456"
            assert user.email == "nested@example.com"
            assert user.first_name == "Jane"
            assert user.last_name == "Smith"
            assert user.image_url == "https://example.com/jane.jpg"

    def test_validate_token_with_metadata(self, config: ClerkConfig) -> None:
        """Test validation extracts metadata from token."""
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        payload = {
            "sub": "user_789",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "public_metadata": {"role": "admin"},
            "private_metadata": {"internal_id": "abc123"},
            "unsafe_metadata": {"custom": "value"},
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            user = validator.validate_token("valid.token.here")

            assert user.metadata["public"] == {"role": "admin"}
            assert user.metadata["private"] == {"internal_id": "abc123"}
            assert user.metadata["unsafe"] == {"custom": "value"}

    def test_validate_token_calls_decode_with_options(self, config: ClerkConfig) -> None:
        """Test validation passes correct options to jwt.decode."""
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        payload = {
            "sub": "user_123",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            validator.validate_token("test.token")

            mock_decode.assert_called_once()
            call_kwargs = mock_decode.call_args

            # Check options
            options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
            assert options["require"] == ["exp", "sub", "iat"]
            assert options["leeway"] == 30

            # Check algorithms
            algorithms = call_kwargs.kwargs.get("algorithms") or call_kwargs[1].get("algorithms")
            assert algorithms == ["RS256"]

    def test_validate_token_with_issuer(self, config: ClerkConfig) -> None:
        """Test validation includes issuer when configured."""
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        payload = {
            "sub": "user_123",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            validator.validate_token("test.token")

            call_kwargs = mock_decode.call_args
            issuer = call_kwargs.kwargs.get("issuer")
            assert issuer == "https://clerk.example.com"

    def test_validate_token_with_audience(self, config_with_audience: ClerkConfig) -> None:
        """Test validation includes audience when configured."""
        validator = ClerkJWTValidator(config_with_audience)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        payload = {
            "sub": "user_123",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            validator.validate_token("test.token")

            call_kwargs = mock_decode.call_args
            audience = call_kwargs.kwargs.get("audience")
            assert audience == "my-app"

    def test_validate_token_without_issuer(self) -> None:
        """Test validation works without issuer (branch coverage)."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="",  # Empty issuer
        )
        validator = ClerkJWTValidator(config)

        mock_key = mock.MagicMock()
        mock_key.key = "fake-key"
        mock_client = mock.MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_key
        validator._jwks_client = mock_client

        payload = {
            "sub": "user_123",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            validator.validate_token("test.token")

            call_kwargs = mock_decode.call_args
            # Should NOT have issuer kwarg since config.issuer is empty
            assert "issuer" not in call_kwargs.kwargs

    def test_decode_without_verification(self, config: ClerkConfig) -> None:
        """Test decode_without_verification for debugging."""
        validator = ClerkJWTValidator(config)

        # Create a simple unsigned token for testing
        payload: dict[str, Any] = {
            "sub": "user_123",
            "email": "user@example.com",
        }

        with mock.patch("rl_emails.api.auth.clerk.jwt.decode") as mock_decode:
            mock_decode.return_value = payload

            result = validator.decode_without_verification("any.token")

            mock_decode.assert_called_once_with("any.token", options={"verify_signature": False})
            assert result == payload
