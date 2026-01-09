# Phase 3: FastAPI + Clerk Authentication & Real-Time Sync

## Status: Planning

**Prerequisite**: Phase 2 Complete (Gmail API Integration)
**Iterations**: 10
**Goal**: Production-ready API with Clerk auth, multi-provider email connections, real-time sync, and mobile-ready endpoints

---

## Overview

Phase 3 transforms the CLI-based pipeline into a production API service with:
- **Clerk Authentication**: Secure user auth with JWT validation
- **Multi-Provider Email**: Gmail now, extensible to Outlook, IMAP, etc.
- **Real-Time Sync**: Webhooks + background workers for fresh data
- **Project/Task Management**: Surface actionable items from emails
- **Mobile-Ready API**: RESTful endpoints optimized for mobile clients

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3 ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Mobile App   │    │   Web App    │    │  Webhooks    │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             │                                           │
│                    ┌────────▼────────┐                                  │
│                    │   FastAPI       │◄─── Clerk JWT Validation         │
│                    │   Gateway       │                                  │
│                    └────────┬────────┘                                  │
│                             │                                           │
│         ┌───────────────────┼───────────────────┐                       │
│         │                   │                   │                       │
│  ┌──────▼──────┐    ┌───────▼──────┐    ┌──────▼──────┐               │
│  │ Auth/Users  │    │ Email Sync   │    │  Projects   │               │
│  │ Service     │    │ Service      │    │  & Tasks    │               │
│  └─────────────┘    └──────┬───────┘    └─────────────┘               │
│                            │                                           │
│         ┌──────────────────┼──────────────────┐                        │
│         │                  │                  │                        │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌─────▼──────┐                 │
│  │   Gmail     │    │   Outlook   │    │    IMAP    │                 │
│  │   Provider  │    │   (Future)  │    │  (Future)  │                 │
│  └─────────────┘    └─────────────┘    └────────────┘                 │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    Background Workers                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │  │
│  │  │ Sync Worker │  │ ML Pipeline │  │ Notification│             │  │
│  │  │ (Webhooks)  │  │   Worker    │  │   Worker    │             │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                      PostgreSQL                                  │  │
│  │   emails | projects | tasks | connections | sync_state          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

1. **Verifiable**: Every feature has clear acceptance criteria and verification steps
2. **Testable**: Unit tests + integration tests with mocked external services
3. **Reliable**: Graceful error handling, retries, circuit breakers
4. **Performant**: Async everywhere, connection pooling, caching where appropriate
5. **Security-driven**: Clerk JWT validation, input sanitization, rate limiting
6. **Well-tested**: 100% coverage on business logic, integration tests for API endpoints

---

## Iteration 1: FastAPI Foundation

### Story
As a developer, I need a FastAPI application structure so that I can build secure, documented API endpoints.

### Deliverables
1. FastAPI application with proper project structure
2. Health check and readiness endpoints
3. OpenAPI documentation configuration
4. CORS middleware for mobile/web clients
5. Error handling middleware
6. Logging configuration

### Architecture

```
src/rl_emails/
├── api/                          # NEW: FastAPI application
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory
│   ├── config.py                 # API-specific config
│   ├── dependencies.py           # Dependency injection
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── cors.py              # CORS configuration
│   │   ├── error_handler.py     # Global error handling
│   │   └── logging.py           # Request logging
│   └── routes/
│       ├── __init__.py
│       └── health.py            # Health check endpoints
```

### Files to Create

| File | Description |
|------|-------------|
| `src/rl_emails/api/__init__.py` | API module exports |
| `src/rl_emails/api/main.py` | FastAPI application factory |
| `src/rl_emails/api/config.py` | API configuration |
| `src/rl_emails/api/dependencies.py` | Dependency injection |
| `src/rl_emails/api/middleware/cors.py` | CORS middleware |
| `src/rl_emails/api/middleware/error_handler.py` | Error handling |
| `src/rl_emails/api/middleware/logging.py` | Request logging |
| `src/rl_emails/api/routes/health.py` | Health endpoints |
| `tests/unit/api/test_main.py` | App factory tests |
| `tests/unit/api/test_health.py` | Health endpoint tests |
| `tests/integration/api/test_app.py` | Integration tests |

### Implementation Design

```python
# src/rl_emails/api/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from rl_emails.api.config import APIConfig
from rl_emails.api.middleware.cors import setup_cors
from rl_emails.api.middleware.error_handler import setup_error_handlers
from rl_emails.api.routes import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await setup_database_pool()
    yield
    # Shutdown
    await close_database_pool()


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    if config is None:
        config = APIConfig.from_env()

    app = FastAPI(
        title="RL-Emails API",
        description="Email ML pipeline API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
    )

    # Middleware
    setup_cors(app, config)
    setup_error_handlers(app)

    # Routes
    app.include_router(health.router, tags=["Health"])

    return app
```

```python
# src/rl_emails/api/routes/health.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.api.dependencies import get_db_session

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="unknown",
    )


@router.get("/health/ready", response_model=HealthResponse)
async def readiness_check(
    session: AsyncSession = Depends(get_db_session),
) -> HealthResponse:
    """Readiness check with database validation."""
    # Verify database connection
    try:
        await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="ready" if db_status == "connected" else "not_ready",
        version="1.0.0",
        database=db_status,
    )
```

### Acceptance Criteria

- [ ] FastAPI app factory creates configured application
- [ ] Health endpoint returns 200 with status info
- [ ] Readiness endpoint validates database connection
- [ ] CORS allows configured origins
- [ ] Global error handler returns consistent error format
- [ ] Request logging captures method, path, duration
- [ ] OpenAPI docs available at /docs
- [ ] App starts with `uvicorn rl_emails.api.main:app`
- [ ] 100% test coverage on new code
- [ ] mypy --strict passes

### Test Plan

```python
# tests/unit/api/test_health.py
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_with_db(self, client, mock_db):
        """Readiness endpoint checks database."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["database"] == "connected"

    def test_readiness_without_db(self, client, mock_db_fail):
        """Readiness returns not_ready when DB fails."""
        response = client.get("/health/ready")
        assert response.json()["status"] == "not_ready"


# tests/integration/api/test_app.py
class TestAPIIntegration:
    """Integration tests for API."""

    @pytest.fixture
    def app(self):
        """Create test application."""
        return create_app(APIConfig(database_url=TEST_DB_URL))

    async def test_full_health_flow(self, app):
        """Test health endpoints with real database."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health/ready")
            assert response.status_code == 200
```

### Verification Steps

1. **Start application**:
   ```bash
   uvicorn rl_emails.api.main:create_app --factory --reload
   ```

2. **Test health endpoints**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/health/ready
   ```

3. **Check OpenAPI docs**:
   - Navigate to http://localhost:8000/docs

4. **Run tests**:
   ```bash
   make check
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| App startup | < 2 seconds |
| Health endpoint | < 10ms response |
| Test coverage | 100% on new code |
| Type safety | mypy --strict passes |

---

## Iteration 2: Clerk Authentication

### Story
As a user, I need to authenticate with Clerk so that I can securely access my email data through the API.

### Deliverables
1. Clerk JWT validation middleware
2. User context extraction from JWT
3. Protected route decorator
4. User sync from Clerk to local database
5. API key support for service-to-service calls

### Architecture

```
src/rl_emails/
├── api/
│   ├── auth/                     # NEW: Authentication module
│   │   ├── __init__.py
│   │   ├── clerk.py             # Clerk JWT validation
│   │   ├── middleware.py        # Auth middleware
│   │   ├── dependencies.py      # Auth dependencies
│   │   └── models.py            # Auth-related models
│   └── routes/
│       └── users.py             # User profile endpoints
```

### Files to Create

| File | Description |
|------|-------------|
| `src/rl_emails/api/auth/__init__.py` | Auth module exports |
| `src/rl_emails/api/auth/clerk.py` | Clerk JWT validation |
| `src/rl_emails/api/auth/middleware.py` | Auth middleware |
| `src/rl_emails/api/auth/dependencies.py` | Auth dependencies |
| `src/rl_emails/api/auth/models.py` | User context models |
| `src/rl_emails/api/routes/users.py` | User endpoints |
| `tests/unit/api/auth/test_clerk.py` | Clerk validation tests |
| `tests/unit/api/auth/test_middleware.py` | Middleware tests |
| `tests/integration/api/test_auth.py` | Auth integration tests |

### Implementation Design

```python
# src/rl_emails/api/auth/clerk.py
from dataclasses import dataclass
from datetime import datetime
import httpx
import jwt
from jwt import PyJWKClient

from rl_emails.api.auth.models import ClerkUser


@dataclass
class ClerkConfig:
    """Clerk configuration."""
    publishable_key: str
    secret_key: str
    jwks_url: str = "https://clerk.dev/.well-known/jwks.json"
    issuer: str | None = None


class ClerkJWTValidator:
    """Validates Clerk JWT tokens."""

    def __init__(self, config: ClerkConfig) -> None:
        self.config = config
        self._jwks_client = PyJWKClient(config.jwks_url)

    async def validate_token(self, token: str) -> ClerkUser:
        """Validate JWT and extract user info.

        Args:
            token: Bearer token from Authorization header.

        Returns:
            ClerkUser with user info.

        Raises:
            AuthenticationError: If token is invalid.
        """
        try:
            # Get signing key
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)

            # Decode and validate
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.config.issuer,
                options={"verify_aud": False},  # Clerk doesn't use aud
            )

            return ClerkUser(
                id=payload["sub"],
                email=payload.get("email"),
                first_name=payload.get("first_name"),
                last_name=payload.get("last_name"),
                image_url=payload.get("image_url"),
                created_at=datetime.fromtimestamp(payload["iat"]),
            )

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")


class AuthenticationError(Exception):
    """Authentication failed."""
    pass
```

```python
# src/rl_emails/api/auth/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from rl_emails.api.auth.clerk import ClerkJWTValidator, AuthenticationError
from rl_emails.api.auth.models import ClerkUser, CurrentUser


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    validator: ClerkJWTValidator = Depends(get_clerk_validator),
) -> CurrentUser:
    """Extract and validate current user from JWT.

    Usage:
        @router.get("/profile")
        async def get_profile(user: CurrentUser = Depends(get_current_user)):
            return {"user_id": user.id}
    """
    try:
        clerk_user = await validator.validate_token(credentials.credentials)

        # Sync user to local database if needed
        local_user = await sync_clerk_user(clerk_user)

        return CurrentUser(
            id=local_user.id,
            clerk_id=clerk_user.id,
            email=clerk_user.email,
            is_authenticated=True,
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
    validator: ClerkJWTValidator = Depends(get_clerk_validator),
) -> CurrentUser | None:
    """Get current user if authenticated, None otherwise."""
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials, validator)
    except HTTPException:
        return None
```

```python
# src/rl_emails/api/auth/models.py
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass
class ClerkUser:
    """User info from Clerk JWT."""
    id: str
    email: str | None
    first_name: str | None
    last_name: str | None
    image_url: str | None
    created_at: datetime


@dataclass
class CurrentUser:
    """Current authenticated user context."""
    id: UUID
    clerk_id: str
    email: str | None
    is_authenticated: bool = True

    @property
    def display_name(self) -> str:
        """Get display name for user."""
        return self.email or self.clerk_id
```

### Acceptance Criteria

- [ ] ClerkJWTValidator validates RS256 JWTs
- [ ] Token expiration is enforced
- [ ] Invalid tokens return 401 with proper message
- [ ] get_current_user dependency extracts user from request
- [ ] User is synced to local database on first auth
- [ ] Protected routes require valid JWT
- [ ] JWKS caching reduces validation latency
- [ ] API key auth works for service accounts
- [ ] 100% test coverage on new code
- [ ] mypy --strict passes

### Test Plan

```python
# tests/unit/api/auth/test_clerk.py
class TestClerkJWTValidator:
    """Tests for Clerk JWT validation."""

    def test_validates_valid_token(self, mock_jwks):
        """Valid token returns ClerkUser."""
        token = create_test_jwt(sub="user_123", email="test@example.com")

        validator = ClerkJWTValidator(config)
        user = await validator.validate_token(token)

        assert user.id == "user_123"
        assert user.email == "test@example.com"

    def test_rejects_expired_token(self, mock_jwks):
        """Expired token raises AuthenticationError."""
        token = create_test_jwt(exp=datetime.now() - timedelta(hours=1))

        validator = ClerkJWTValidator(config)
        with pytest.raises(AuthenticationError, match="expired"):
            await validator.validate_token(token)

    def test_rejects_invalid_signature(self, mock_jwks):
        """Invalid signature raises AuthenticationError."""
        token = "invalid.token.here"

        validator = ClerkJWTValidator(config)
        with pytest.raises(AuthenticationError, match="Invalid token"):
            await validator.validate_token(token)


# tests/integration/api/test_auth.py
class TestAuthIntegration:
    """Integration tests for authentication."""

    async def test_protected_route_requires_auth(self, client):
        """Protected route returns 401 without token."""
        response = await client.get("/api/v1/profile")
        assert response.status_code == 401

    async def test_protected_route_with_valid_token(self, client, mock_clerk):
        """Protected route succeeds with valid token."""
        token = mock_clerk.create_token(user_id="user_123")

        response = await client.get(
            "/api/v1/profile",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
```

### Verification Steps

1. **Configure Clerk**:
   ```bash
   # Add to .env
   CLERK_PUBLISHABLE_KEY=pk_test_...
   CLERK_SECRET_KEY=sk_test_...
   ```

2. **Test with Clerk token**:
   ```bash
   # Get token from Clerk frontend
   curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/profile
   ```

3. **Test without token**:
   ```bash
   curl http://localhost:8000/api/v1/profile
   # Should return 401
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| JWT validation | < 50ms (with JWKS cache) |
| Auth failure | Clear error messages |
| User sync | Automatic on first auth |
| Test coverage | 100% on new code |

---

## Iteration 3: Email Connection Provider Interface

### Story
As a developer, I need a provider interface so that I can support multiple email services (Gmail, Outlook, IMAP) with a unified API.

### Deliverables
1. Email provider abstract interface
2. Gmail provider implementation (wrap existing)
3. Connection management service
4. Provider registry pattern
5. Connection status endpoints

### Architecture

```
src/rl_emails/
├── providers/                    # NEW: Email provider abstraction
│   ├── __init__.py
│   ├── base.py                  # Abstract provider interface
│   ├── registry.py              # Provider registry
│   ├── gmail/                   # Gmail provider
│   │   ├── __init__.py
│   │   └── provider.py
│   └── types.py                 # Shared types
├── services/
│   └── connection_service.py    # NEW: Connection management
└── api/routes/
    └── connections.py           # NEW: Connection endpoints
```

### Implementation Design

```python
# src/rl_emails/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import AsyncIterator
from uuid import UUID


class ProviderType(str, Enum):
    """Supported email providers."""
    GMAIL = "gmail"
    OUTLOOK = "outlook"  # Future
    IMAP = "imap"        # Future


@dataclass
class ConnectionStatus:
    """Status of an email connection."""
    connected: bool
    provider: ProviderType
    email: str
    last_sync: datetime | None
    error: str | None


@dataclass
class EmailMessage:
    """Unified email message format."""
    id: str
    provider_id: str  # Gmail ID, Outlook ID, etc.
    thread_id: str | None
    message_id: str  # RFC 5322 Message-ID
    subject: str
    from_email: str
    from_name: str | None
    to_emails: list[str]
    cc_emails: list[str]
    date: datetime
    body_text: str
    body_html: str | None
    labels: list[str]
    has_attachments: bool


class EmailProvider(ABC):
    """Abstract email provider interface.

    All email providers (Gmail, Outlook, IMAP) implement this interface.
    """

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return provider type identifier."""
        pass

    @abstractmethod
    async def connect(self, user_id: UUID, credentials: dict) -> ConnectionStatus:
        """Establish connection with provider.

        Args:
            user_id: User to connect for.
            credentials: Provider-specific credentials.

        Returns:
            ConnectionStatus indicating success/failure.
        """
        pass

    @abstractmethod
    async def disconnect(self, user_id: UUID) -> None:
        """Disconnect and revoke access."""
        pass

    @abstractmethod
    async def get_status(self, user_id: UUID) -> ConnectionStatus:
        """Get current connection status."""
        pass

    @abstractmethod
    async def sync_messages(
        self,
        user_id: UUID,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EmailMessage]:
        """Sync messages from provider.

        Args:
            user_id: User to sync for.
            since: Only sync messages after this date.
            limit: Maximum messages to sync.

        Yields:
            EmailMessage objects.
        """
        pass

    @abstractmethod
    async def get_auth_url(self, user_id: UUID, redirect_uri: str) -> str:
        """Get OAuth authorization URL.

        For OAuth providers (Gmail, Outlook).
        """
        pass

    @abstractmethod
    async def handle_oauth_callback(
        self,
        user_id: UUID,
        code: str,
        redirect_uri: str,
    ) -> ConnectionStatus:
        """Handle OAuth callback.

        Exchange authorization code for tokens and store connection.
        """
        pass
```

```python
# src/rl_emails/providers/gmail/provider.py
from rl_emails.providers.base import (
    EmailProvider,
    ProviderType,
    ConnectionStatus,
    EmailMessage,
)
from rl_emails.integrations.gmail.client import GmailClient
from rl_emails.services.auth_service import AuthService


class GmailProvider(EmailProvider):
    """Gmail implementation of email provider."""

    def __init__(
        self,
        auth_service: AuthService,
        sync_service: SyncService,
    ) -> None:
        self._auth_service = auth_service
        self._sync_service = sync_service

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GMAIL

    async def connect(self, user_id: UUID, credentials: dict) -> ConnectionStatus:
        """Connect Gmail account."""
        # Use existing AuthService
        code = credentials.get("code")
        if not code:
            raise ValueError("Missing authorization code")

        await self._auth_service.complete_auth_flow(user_id, code)
        return await self.get_status(user_id)

    async def sync_messages(
        self,
        user_id: UUID,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EmailMessage]:
        """Sync Gmail messages."""
        # Get valid access token
        token = await self._auth_service.get_valid_token(user_id)

        async with GmailClient(access_token=token) as client:
            # Use existing sync logic
            days = (datetime.now() - since).days if since else 30

            async for gmail_msg in client.list_all_messages(
                query=f"newer_than:{days}d",
                max_messages=limit,
            ):
                full_msg = await client.get_message(gmail_msg.id)
                yield self._convert_message(full_msg)

    def _convert_message(self, gmail_msg: GmailMessage) -> EmailMessage:
        """Convert Gmail message to unified format."""
        return EmailMessage(
            id=gmail_msg.id,
            provider_id=gmail_msg.id,
            thread_id=gmail_msg.thread_id,
            message_id=gmail_msg.message_id,
            subject=gmail_msg.subject,
            from_email=gmail_msg.from_address,
            from_name=None,  # Parse from from_address
            to_emails=gmail_msg.to_addresses,
            cc_emails=gmail_msg.cc_addresses,
            date=gmail_msg.date_sent,
            body_text=gmail_msg.body_plain or "",
            body_html=gmail_msg.body_html,
            labels=gmail_msg.label_ids,
            has_attachments=gmail_msg.has_attachments,
        )
```

```python
# src/rl_emails/providers/registry.py
from rl_emails.providers.base import EmailProvider, ProviderType


class ProviderRegistry:
    """Registry for email providers."""

    def __init__(self) -> None:
        self._providers: dict[ProviderType, EmailProvider] = {}

    def register(self, provider: EmailProvider) -> None:
        """Register a provider."""
        self._providers[provider.provider_type] = provider

    def get(self, provider_type: ProviderType) -> EmailProvider:
        """Get provider by type."""
        if provider_type not in self._providers:
            raise ValueError(f"Unknown provider: {provider_type}")
        return self._providers[provider_type]

    def list_providers(self) -> list[ProviderType]:
        """List all registered providers."""
        return list(self._providers.keys())
```

### Acceptance Criteria

- [ ] EmailProvider abstract class defines clear interface
- [ ] GmailProvider implements all EmailProvider methods
- [ ] Provider registry supports dynamic registration
- [ ] Connection service manages multi-provider connections
- [ ] API endpoints for connect/disconnect/status
- [ ] Unified EmailMessage format works across providers
- [ ] 100% test coverage on new code
- [ ] mypy --strict passes

---

## Iteration 4: Real-Time Sync with Gmail Push Notifications

### Story
As a user, I need real-time email updates so that my data is always fresh without manual sync.

### Deliverables
1. Gmail Push Notification (Pub/Sub) integration
2. Webhook endpoint for Gmail notifications
3. Background sync worker
4. Sync state management
5. Notification deduplication

### Architecture

```
src/rl_emails/
├── api/routes/
│   └── webhooks.py              # NEW: Webhook endpoints
├── workers/                      # NEW: Background workers
│   ├── __init__.py
│   ├── sync_worker.py           # Sync processing worker
│   └── scheduler.py             # Cron-like scheduler
└── services/
    └── push_notification.py     # NEW: Push notification handling
```

### Implementation Design

```python
# src/rl_emails/services/push_notification.py
from dataclasses import dataclass
from uuid import UUID
import base64
import json


@dataclass
class GmailPushNotification:
    """Gmail push notification payload."""
    email_address: str
    history_id: str


class PushNotificationService:
    """Handles Gmail push notifications."""

    async def setup_watch(self, user_id: UUID) -> str:
        """Setup Gmail push notifications for user.

        Returns:
            Expiration timestamp for the watch.
        """
        token = await self._auth_service.get_valid_token(user_id)

        async with GmailClient(access_token=token) as client:
            result = await client.setup_watch(
                topic_name=self._config.pubsub_topic,
                label_ids=["INBOX"],
            )

            # Store watch expiration
            await self._sync_repo.update_watch(
                user_id,
                expiration=result["expiration"],
                history_id=result["historyId"],
            )

            return result["expiration"]

    async def handle_notification(
        self,
        message_data: str,
    ) -> None:
        """Handle incoming push notification.

        Args:
            message_data: Base64-encoded notification payload.
        """
        # Decode payload
        payload = json.loads(base64.urlsafe_b64decode(message_data))
        notification = GmailPushNotification(
            email_address=payload["emailAddress"],
            history_id=payload["historyId"],
        )

        # Find user by email
        user = await self._user_repo.get_by_email(notification.email_address)
        if not user:
            return

        # Queue incremental sync
        await self._sync_queue.enqueue(
            user_id=user.id,
            history_id=notification.history_id,
        )
```

```python
# src/rl_emails/api/routes/webhooks.py
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.post("/webhooks/gmail")
async def gmail_webhook(
    request: Request,
    push_service: PushNotificationService = Depends(get_push_service),
) -> dict:
    """Handle Gmail push notification webhook.

    This endpoint receives notifications from Google Pub/Sub
    when emails are added/modified/deleted.
    """
    body = await request.json()

    # Verify it's from Google (check subscription)
    if not verify_pubsub_token(request):
        raise HTTPException(status_code=403, detail="Invalid token")

    # Handle notification
    message_data = body.get("message", {}).get("data")
    if message_data:
        await push_service.handle_notification(message_data)

    return {"status": "ok"}
```

```python
# src/rl_emails/workers/sync_worker.py
import asyncio
from datetime import datetime


class SyncWorker:
    """Background worker for email sync processing."""

    async def run(self) -> None:
        """Run the sync worker loop."""
        while True:
            try:
                # Process queued sync jobs
                job = await self._queue.dequeue()
                if job:
                    await self._process_sync(job)
                else:
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                await asyncio.sleep(5)

    async def _process_sync(self, job: SyncJob) -> None:
        """Process a sync job."""
        user_id = job.user_id

        # Get provider
        connection = await self._connection_service.get_connection(user_id)
        provider = self._registry.get(connection.provider_type)

        # Run incremental sync
        async for message in provider.sync_messages(
            user_id,
            since=job.since,
        ):
            await self._store_message(user_id, message)

        # Run pipeline stages
        await self._batch_processor.process_batch(
            user_id,
            run_embeddings=True,
            run_llm=True,
        )
```

### Acceptance Criteria

- [ ] Gmail watch setup creates Pub/Sub subscription
- [ ] Webhook endpoint receives and validates notifications
- [ ] Sync worker processes queued sync jobs
- [ ] Incremental sync uses history_id
- [ ] Duplicate notifications are deduplicated
- [ ] Watch expiration is monitored and renewed
- [ ] Sync errors don't crash the worker
- [ ] 100% test coverage on new code

---

## Iteration 5: Projects & Tasks Extraction

### Story
As a user, I need to see projects and tasks extracted from my emails so that I can focus on what needs attention.

### Deliverables
1. Project detection from email patterns
2. Task extraction from email content
3. Project/Task database models
4. API endpoints for projects and tasks
5. Task prioritization based on email priority

### Architecture

```
src/rl_emails/
├── models/
│   ├── project.py               # NEW: Project model
│   └── task.py                  # NEW: Task model
├── services/
│   ├── project_service.py       # NEW: Project management
│   └── task_service.py          # NEW: Task extraction
└── api/routes/
    ├── projects.py              # NEW: Project endpoints
    └── tasks.py                 # NEW: Task endpoints
```

### Implementation Design

```python
# src/rl_emails/models/project.py
from sqlalchemy import Column, String, ForeignKey, DateTime, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from rl_emails.models.base import Base


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Project(Base):
    """Email-derived project."""
    __tablename__ = "projects"

    id = Column(UUID, primary_key=True)
    user_id = Column(UUID, ForeignKey("org_users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String)
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE)

    # Detection metadata
    detected_from = Column(String)  # "cluster", "subject_pattern", "manual"
    cluster_id = Column(Integer, ForeignKey("email_clusters.id"))

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Relationships
    tasks = relationship("Task", back_populates="project")
    emails = relationship("Email", secondary="project_emails")
```

```python
# src/rl_emails/services/project_service.py
from uuid import UUID


class ProjectService:
    """Service for project detection and management."""

    async def detect_projects(self, user_id: UUID) -> list[Project]:
        """Detect projects from email clusters.

        Uses cluster metadata and email patterns to identify projects.
        """
        # Get clusters with project-like characteristics
        clusters = await self._cluster_repo.get_project_clusters(user_id)

        projects = []
        for cluster in clusters:
            # Check if project already exists
            existing = await self._project_repo.get_by_cluster(cluster.id)
            if existing:
                continue

            # Create project from cluster
            project = Project(
                user_id=user_id,
                name=self._generate_project_name(cluster),
                description=cluster.summary,
                detected_from="cluster",
                cluster_id=cluster.id,
            )

            await self._project_repo.create(project)
            projects.append(project)

        return projects

    def _generate_project_name(self, cluster: EmailCluster) -> str:
        """Generate project name from cluster."""
        # Use common subject terms
        if cluster.common_subject:
            return cluster.common_subject

        # Use sender domain
        if cluster.primary_sender:
            return f"Project: {cluster.primary_sender}"

        return f"Project #{cluster.id}"
```

```python
# src/rl_emails/services/task_service.py
class TaskService:
    """Service for task extraction and management."""

    async def extract_tasks(self, user_id: UUID, email_id: int) -> list[Task]:
        """Extract tasks from an email.

        Uses LLM classification and pattern matching to identify actionable items.
        """
        email = await self._email_repo.get(email_id)
        if not email:
            return []

        # Get LLM classification
        classification = await self._llm_repo.get_by_email(email_id)
        if not classification:
            return []

        tasks = []

        # Check if email requires action
        if classification.action_required:
            task = Task(
                user_id=user_id,
                email_id=email_id,
                title=self._generate_task_title(email, classification),
                description=classification.action_summary,
                due_date=classification.suggested_deadline,
                priority=self._map_priority(email.priority_score),
            )
            tasks.append(task)

        # Extract sub-tasks from action items
        for action_item in classification.action_items or []:
            sub_task = Task(
                user_id=user_id,
                email_id=email_id,
                parent_task_id=task.id if tasks else None,
                title=action_item,
                priority=TaskPriority.MEDIUM,
            )
            tasks.append(sub_task)

        return tasks
```

### Acceptance Criteria

- [ ] Project model with cluster association
- [ ] Task model with email association
- [ ] Project detection from clusters
- [ ] Task extraction from LLM classification
- [ ] API endpoints for CRUD operations
- [ ] Projects linked to relevant emails
- [ ] Tasks sorted by priority
- [ ] 100% test coverage on new code

---

## Iteration 6: Mobile-Optimized API Endpoints

### Story
As a mobile developer, I need optimized API endpoints so that the mobile app can efficiently fetch and display email data.

### Deliverables
1. Paginated email list endpoint
2. Email detail endpoint with thread context
3. Priority inbox endpoint
4. Search endpoint with filters
5. Bulk operations (mark read, archive)
6. Response compression

### Architecture

```
src/rl_emails/
├── api/routes/
│   ├── emails.py                # NEW: Email endpoints
│   ├── inbox.py                 # NEW: Inbox/priority endpoints
│   └── search.py                # NEW: Search endpoint
└── api/
    └── schemas/                 # NEW: Pydantic response schemas
        ├── __init__.py
        ├── email.py
        ├── inbox.py
        └── pagination.py
```

### Implementation Design

```python
# src/rl_emails/api/schemas/email.py
from pydantic import BaseModel
from datetime import datetime


class EmailListItem(BaseModel):
    """Email in list view (minimal data)."""
    id: int
    subject: str
    from_email: str
    from_name: str | None
    preview: str  # First 100 chars
    date: datetime
    is_read: bool
    is_starred: bool
    has_attachments: bool
    priority_score: float
    labels: list[str]


class EmailDetail(BaseModel):
    """Full email detail."""
    id: int
    subject: str
    from_email: str
    from_name: str | None
    to_emails: list[str]
    cc_emails: list[str]
    date: datetime
    body_text: str
    body_html: str | None
    is_read: bool
    is_starred: bool
    has_attachments: bool
    attachments: list[AttachmentInfo]
    priority_score: float
    labels: list[str]
    thread: list[EmailListItem] | None

    # AI-generated
    summary: str | None
    action_required: bool
    action_items: list[str]


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool
```

```python
# src/rl_emails/api/routes/emails.py
from fastapi import APIRouter, Depends, Query

router = APIRouter()


@router.get("/emails", response_model=PaginatedResponse[EmailListItem])
async def list_emails(
    user: CurrentUser = Depends(get_current_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    label: str | None = None,
    is_read: bool | None = None,
    sort_by: str = Query("date", pattern="^(date|priority|from)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    email_service: EmailService = Depends(get_email_service),
) -> PaginatedResponse[EmailListItem]:
    """List emails with pagination and filters."""
    result = await email_service.list_emails(
        user_id=user.id,
        page=page,
        page_size=page_size,
        filters=EmailFilters(label=label, is_read=is_read),
        sort_by=sort_by,
        order=order,
    )

    return PaginatedResponse(
        items=result.items,
        total=result.total,
        page=page,
        page_size=page_size,
        has_next=page * page_size < result.total,
        has_prev=page > 1,
    )


@router.get("/emails/{email_id}", response_model=EmailDetail)
async def get_email(
    email_id: int,
    user: CurrentUser = Depends(get_current_user),
    include_thread: bool = Query(True),
    email_service: EmailService = Depends(get_email_service),
) -> EmailDetail:
    """Get full email detail with optional thread context."""
    email = await email_service.get_email(
        user_id=user.id,
        email_id=email_id,
        include_thread=include_thread,
    )

    if not email:
        raise HTTPException(status_code=404, detail="Email not found")

    return email


@router.get("/inbox/priority", response_model=PaginatedResponse[EmailListItem])
async def get_priority_inbox(
    user: CurrentUser = Depends(get_current_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    email_service: EmailService = Depends(get_email_service),
) -> PaginatedResponse[EmailListItem]:
    """Get priority-sorted inbox.

    Returns emails sorted by ML-computed priority score.
    """
    return await email_service.get_priority_inbox(
        user_id=user.id,
        page=page,
        page_size=page_size,
    )


@router.post("/emails/bulk/mark-read")
async def bulk_mark_read(
    request: BulkMarkReadRequest,
    user: CurrentUser = Depends(get_current_user),
    email_service: EmailService = Depends(get_email_service),
) -> dict:
    """Mark multiple emails as read."""
    count = await email_service.bulk_mark_read(
        user_id=user.id,
        email_ids=request.email_ids,
    )
    return {"updated": count}
```

### Acceptance Criteria

- [ ] Email list with pagination and filters
- [ ] Email detail with full content
- [ ] Thread context included in detail
- [ ] Priority inbox sorts by ML score
- [ ] Search with full-text and filters
- [ ] Bulk operations for efficiency
- [ ] Gzip compression on large responses
- [ ] Response times < 200ms
- [ ] 100% test coverage on new code

---

## Iteration 7: Background Pipeline Worker

### Story
As a system, I need background workers to process emails through the ML pipeline so that classification and priority scores are always up to date.

### Deliverables
1. Pipeline worker service
2. Job queue with Redis/PostgreSQL
3. Worker health monitoring
4. Batch processing optimization
5. Error handling and retries

### Architecture

```
src/rl_emails/
├── workers/
│   ├── pipeline_worker.py       # NEW: ML pipeline worker
│   ├── queue.py                 # NEW: Job queue abstraction
│   └── monitor.py               # NEW: Worker monitoring
```

### Implementation Design

```python
# src/rl_emails/workers/pipeline_worker.py
import asyncio
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PipelineJob:
    """Job for pipeline processing."""
    id: str
    user_id: UUID
    email_ids: list[int]
    stages: list[str]
    created_at: datetime
    priority: int = 0


class PipelineWorker:
    """Background worker for ML pipeline processing."""

    def __init__(
        self,
        queue: JobQueue,
        batch_processor: BatchProcessor,
        config: WorkerConfig,
    ) -> None:
        self._queue = queue
        self._processor = batch_processor
        self._config = config
        self._running = False

    async def start(self) -> None:
        """Start the worker."""
        self._running = True
        logger.info("Pipeline worker started")

        while self._running:
            try:
                job = await self._queue.dequeue(timeout=5.0)
                if job:
                    await self._process_job(job)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False
        logger.info("Pipeline worker stopping")

    async def _process_job(self, job: PipelineJob) -> None:
        """Process a pipeline job."""
        logger.info(f"Processing job {job.id}: {len(job.email_ids)} emails")

        try:
            # Get emails to process
            emails = await self._get_emails(job.email_ids)

            # Run requested stages
            result = await self._processor.process_batch(
                emails=emails,
                run_embeddings="embeddings" in job.stages,
                run_llm="llm" in job.stages,
            )

            # Mark job complete
            await self._queue.complete(job.id, result)

            logger.info(f"Job {job.id} complete: {result}")

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            await self._queue.fail(job.id, str(e))

            # Retry logic
            if job.retry_count < self._config.max_retries:
                await self._queue.retry(job)
```

```python
# src/rl_emails/workers/queue.py
from abc import ABC, abstractmethod


class JobQueue(ABC):
    """Abstract job queue interface."""

    @abstractmethod
    async def enqueue(self, job: PipelineJob) -> str:
        """Add job to queue, return job ID."""
        pass

    @abstractmethod
    async def dequeue(self, timeout: float = 0) -> PipelineJob | None:
        """Get next job from queue."""
        pass

    @abstractmethod
    async def complete(self, job_id: str, result: dict) -> None:
        """Mark job as complete."""
        pass

    @abstractmethod
    async def fail(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        pass


class PostgresJobQueue(JobQueue):
    """PostgreSQL-backed job queue."""

    async def enqueue(self, job: PipelineJob) -> str:
        """Add job to queue."""
        result = await self._session.execute(
            text("""
                INSERT INTO pipeline_jobs (user_id, email_ids, stages, priority)
                VALUES (:user_id, :email_ids, :stages, :priority)
                RETURNING id
            """),
            {
                "user_id": str(job.user_id),
                "email_ids": job.email_ids,
                "stages": job.stages,
                "priority": job.priority,
            },
        )
        return str(result.scalar())

    async def dequeue(self, timeout: float = 0) -> PipelineJob | None:
        """Get next job using SELECT FOR UPDATE SKIP LOCKED."""
        result = await self._session.execute(
            text("""
                UPDATE pipeline_jobs
                SET status = 'processing', started_at = NOW()
                WHERE id = (
                    SELECT id FROM pipeline_jobs
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING id, user_id, email_ids, stages
            """)
        )
        row = result.fetchone()
        if not row:
            return None

        return PipelineJob(
            id=str(row.id),
            user_id=UUID(row.user_id),
            email_ids=row.email_ids,
            stages=row.stages,
            created_at=datetime.now(),
        )
```

### Acceptance Criteria

- [ ] Worker processes jobs from queue
- [ ] PostgreSQL queue with SKIP LOCKED
- [ ] Job retry with exponential backoff
- [ ] Worker health endpoint
- [ ] Graceful shutdown
- [ ] Batch processing optimized
- [ ] Job metrics/monitoring
- [ ] 100% test coverage on new code

---

## Iteration 8: IMAP Provider Implementation

### Story
As a user with non-Gmail email, I need IMAP support so that I can connect any email provider.

### Deliverables
1. IMAP provider implementation
2. IMAP connection management
3. IDLE command for real-time updates
4. Credential storage (encrypted)
5. Provider-agnostic sync

### Architecture

```
src/rl_emails/
├── providers/
│   └── imap/                    # NEW: IMAP provider
│       ├── __init__.py
│       ├── provider.py
│       ├── client.py
│       └── idle.py
```

### Implementation Design

```python
# src/rl_emails/providers/imap/provider.py
from rl_emails.providers.base import EmailProvider, ProviderType


class IMAPProvider(EmailProvider):
    """IMAP implementation of email provider."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.IMAP

    async def connect(self, user_id: UUID, credentials: dict) -> ConnectionStatus:
        """Connect IMAP account.

        Args:
            user_id: User ID.
            credentials: Dict with host, port, username, password.
        """
        host = credentials["host"]
        port = credentials.get("port", 993)
        username = credentials["username"]
        password = credentials["password"]
        use_ssl = credentials.get("use_ssl", True)

        # Test connection
        client = IMAPClient(host, port, use_ssl=use_ssl)
        try:
            await client.connect()
            await client.login(username, password)

            # Store encrypted credentials
            await self._store_credentials(user_id, credentials)

            return ConnectionStatus(
                connected=True,
                provider=ProviderType.IMAP,
                email=username,
                last_sync=None,
                error=None,
            )

        except Exception as e:
            return ConnectionStatus(
                connected=False,
                provider=ProviderType.IMAP,
                email=username,
                last_sync=None,
                error=str(e),
            )
        finally:
            await client.disconnect()

    async def sync_messages(
        self,
        user_id: UUID,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EmailMessage]:
        """Sync messages via IMAP."""
        credentials = await self._get_credentials(user_id)

        async with IMAPClient(credentials["host"]) as client:
            await client.login(credentials["username"], credentials["password"])
            await client.select("INBOX")

            # Search for messages
            search_criteria = ["ALL"]
            if since:
                search_criteria = ["SINCE", since.strftime("%d-%b-%Y")]

            message_ids = await client.search(search_criteria)

            if limit:
                message_ids = message_ids[:limit]

            for msg_id in message_ids:
                raw_msg = await client.fetch(msg_id, ["RFC822"])
                yield self._parse_message(raw_msg)
```

### Acceptance Criteria

- [ ] IMAP provider implements EmailProvider interface
- [ ] Connection with SSL/TLS support
- [ ] Credential encryption at rest
- [ ] IDLE support for real-time updates
- [ ] Standard folders (INBOX, SENT, etc.)
- [ ] Message parsing to unified format
- [ ] 100% test coverage on new code

---

## Iteration 9: Scheduled Sync & Watch Renewal

### Story
As a system administrator, I need scheduled tasks to maintain sync freshness and watch renewals.

### Deliverables
1. Scheduled task framework
2. Gmail watch renewal job
3. Periodic full sync job
4. Stale connection detection
5. Admin monitoring endpoints

### Architecture

```
src/rl_emails/
├── workers/
│   ├── scheduler.py             # Scheduled task runner
│   └── jobs/
│       ├── __init__.py
│       ├── watch_renewal.py     # Gmail watch renewal
│       ├── periodic_sync.py     # Full sync job
│       └── stale_detection.py   # Connection health
└── api/routes/
    └── admin.py                 # Admin endpoints
```

### Implementation Design

```python
# src/rl_emails/workers/scheduler.py
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ScheduledJob:
    """Scheduled job configuration."""
    name: str
    func: Callable
    interval: timedelta
    last_run: datetime | None = None


class Scheduler:
    """Simple async scheduler for periodic tasks."""

    def __init__(self) -> None:
        self._jobs: list[ScheduledJob] = []
        self._running = False

    def add_job(
        self,
        name: str,
        func: Callable,
        interval: timedelta,
    ) -> None:
        """Add a scheduled job."""
        self._jobs.append(ScheduledJob(
            name=name,
            func=func,
            interval=interval,
        ))

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True

        while self._running:
            now = datetime.now()

            for job in self._jobs:
                if self._should_run(job, now):
                    try:
                        await job.func()
                        job.last_run = now
                    except Exception as e:
                        logger.error(f"Job {job.name} failed: {e}")

            await asyncio.sleep(60)  # Check every minute

    def _should_run(self, job: ScheduledJob, now: datetime) -> bool:
        """Check if job should run."""
        if job.last_run is None:
            return True
        return now - job.last_run >= job.interval
```

```python
# src/rl_emails/workers/jobs/watch_renewal.py
async def renew_gmail_watches() -> None:
    """Renew expiring Gmail push notification watches.

    Gmail watches expire after 7 days. This job renews watches
    that will expire within 24 hours.
    """
    expiring_soon = datetime.now() + timedelta(hours=24)

    # Get users with expiring watches
    users = await sync_repo.get_users_with_expiring_watch(expiring_soon)

    for user in users:
        try:
            provider = registry.get(ProviderType.GMAIL)
            await provider.setup_watch(user.id)
            logger.info(f"Renewed watch for user {user.id}")
        except Exception as e:
            logger.error(f"Failed to renew watch for {user.id}: {e}")
```

### Acceptance Criteria

- [ ] Scheduler runs periodic tasks
- [ ] Gmail watch renewal before expiration
- [ ] Periodic full sync for consistency
- [ ] Stale connection detection and alerting
- [ ] Admin endpoint for job status
- [ ] Job failure handling and logging
- [ ] 100% test coverage on new code

---

## Iteration 10: Integration Testing & Documentation

### Story
As a team, we need comprehensive integration tests and documentation so that the system is production-ready.

### Deliverables
1. End-to-end integration tests
2. API documentation (OpenAPI + guides)
3. Deployment documentation
4. Performance benchmarks
5. Security audit checklist

### Test Scenarios

```python
# tests/integration/test_full_flow.py
class TestFullUserFlow:
    """End-to-end integration tests."""

    async def test_user_onboarding_flow(self, client, mock_clerk, mock_gmail):
        """Test complete user onboarding."""
        # 1. Authenticate with Clerk
        token = mock_clerk.create_token(user_id="user_123")

        # 2. Connect Gmail
        auth_url = await client.post(
            "/api/v1/connections/gmail/auth-url",
            headers={"Authorization": f"Bearer {token}"},
        )

        # 3. Complete OAuth
        await client.post(
            "/api/v1/connections/gmail/callback",
            json={"code": "mock_code"},
            headers={"Authorization": f"Bearer {token}"},
        )

        # 4. Verify connection
        status = await client.get(
            "/api/v1/connections/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert status.json()["gmail"]["connected"] is True

        # 5. Trigger sync
        await client.post(
            "/api/v1/sync/start",
            headers={"Authorization": f"Bearer {token}"},
        )

        # 6. Wait for sync completion
        await wait_for_sync_complete(client, token)

        # 7. Verify emails in inbox
        inbox = await client.get(
            "/api/v1/inbox/priority",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert len(inbox.json()["items"]) > 0

        # 8. Verify projects detected
        projects = await client.get(
            "/api/v1/projects",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert len(projects.json()["items"]) >= 0

    async def test_real_time_sync(self, client, mock_clerk, mock_gmail):
        """Test real-time webhook sync."""
        token = mock_clerk.create_token(user_id="user_123")

        # Setup: Connect Gmail
        await setup_gmail_connection(client, token)

        # Simulate webhook notification
        await client.post(
            "/webhooks/gmail",
            json=create_gmail_notification("user@gmail.com", "12345"),
        )

        # Verify new email appears
        await asyncio.sleep(2)  # Wait for processing

        inbox = await client.get(
            "/api/v1/inbox",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should include new email
```

### Acceptance Criteria

- [ ] Integration tests cover all user flows
- [ ] API documentation complete and accurate
- [ ] Deployment guide with Docker/K8s examples
- [ ] Performance benchmarks documented
- [ ] Security checklist completed
- [ ] Error scenarios tested
- [ ] Load testing results acceptable
- [ ] 100% coverage on business logic

---

## Phase 3 Completion Checklist

### Pre-Implementation
- [ ] Clerk account and credentials configured
- [ ] Google Cloud Pub/Sub topic created (for webhooks)
- [ ] Redis or PostgreSQL queue ready
- [ ] Phase 2 complete (Gmail API working)

### Iteration 1: FastAPI Foundation
- [ ] Create API module structure
- [ ] Implement app factory
- [ ] Add health endpoints
- [ ] Configure CORS and middleware
- [ ] Write tests

### Iteration 2: Clerk Authentication
- [ ] Implement JWT validation
- [ ] Create auth dependencies
- [ ] Add user sync
- [ ] Write tests

### Iteration 3: Provider Interface
- [ ] Define abstract interface
- [ ] Implement Gmail provider
- [ ] Create provider registry
- [ ] Write tests

### Iteration 4: Real-Time Sync
- [ ] Gmail push notifications
- [ ] Webhook endpoint
- [ ] Sync worker
- [ ] Write tests

### Iteration 5: Projects & Tasks
- [ ] Database models
- [ ] Detection services
- [ ] API endpoints
- [ ] Write tests

### Iteration 6: Mobile API
- [ ] Email list/detail endpoints
- [ ] Priority inbox
- [ ] Search and bulk operations
- [ ] Write tests

### Iteration 7: Pipeline Worker
- [ ] Job queue implementation
- [ ] Pipeline worker
- [ ] Monitoring
- [ ] Write tests

### Iteration 8: IMAP Provider
- [ ] IMAP client
- [ ] Provider implementation
- [ ] IDLE support
- [ ] Write tests

### Iteration 9: Scheduler
- [ ] Scheduled task framework
- [ ] Watch renewal job
- [ ] Periodic sync job
- [ ] Write tests

### Iteration 10: Integration & Docs
- [ ] Integration tests
- [ ] API documentation
- [ ] Deployment guide
- [ ] Performance benchmarks

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing ...

    # FastAPI (new)
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",

    # Authentication
    "pyjwt>=2.8.0",
    "cryptography>=42.0.0",

    # IMAP
    "aioimaplib>=1.0.0",

    # Background workers
    "arq>=0.26.0",  # Or use PostgreSQL queue
]
```

---

## Success Criteria Summary

| Iteration | Key Deliverable | Verification |
|-----------|-----------------|--------------|
| 1 | FastAPI app | Health endpoint works |
| 2 | Clerk auth | Protected routes work |
| 3 | Provider interface | Gmail provider works |
| 4 | Real-time sync | Webhooks trigger sync |
| 5 | Projects/Tasks | Detection works |
| 6 | Mobile API | All endpoints work |
| 7 | Pipeline worker | Background processing works |
| 8 | IMAP provider | Non-Gmail email works |
| 9 | Scheduler | Watches renewed |
| 10 | Integration tests | All scenarios pass |

**Phase 3 Complete When**:
- API serves mobile/web clients
- Clerk authentication working
- Gmail and IMAP providers functional
- Real-time sync via webhooks
- Projects and tasks extracted
- Background workers processing
- 100% test coverage maintained
- Performance benchmarks met
