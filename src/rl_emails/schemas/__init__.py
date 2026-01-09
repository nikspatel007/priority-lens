"""Pydantic schemas for rl-emails."""

from rl_emails.schemas.oauth_token import (
    OAuthTokenCreate,
    OAuthTokenResponse,
    OAuthTokenStatus,
    OAuthTokenUpdate,
)
from rl_emails.schemas.org_user import (
    OrgUserCreate,
    OrgUserResponse,
    OrgUserUpdate,
)
from rl_emails.schemas.organization import (
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
)
from rl_emails.schemas.sync import SyncStateResponse, SyncStateUpdate
from rl_emails.schemas.watch import (
    WatchSubscriptionCreate,
    WatchSubscriptionResponse,
    WatchSubscriptionStatus,
    WatchSubscriptionUpdate,
)

__all__ = [
    "OAuthTokenCreate",
    "OAuthTokenResponse",
    "OAuthTokenStatus",
    "OAuthTokenUpdate",
    "OrganizationCreate",
    "OrganizationResponse",
    "OrganizationUpdate",
    "OrgUserCreate",
    "OrgUserResponse",
    "OrgUserUpdate",
    "SyncStateResponse",
    "SyncStateUpdate",
    "WatchSubscriptionCreate",
    "WatchSubscriptionResponse",
    "WatchSubscriptionStatus",
    "WatchSubscriptionUpdate",
]
