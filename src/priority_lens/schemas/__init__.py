"""Pydantic schemas for rl-emails."""

from priority_lens.schemas.cluster_metadata import (
    ClusterDimension,
    ClusterLabelRequest,
    ClusterLabelResponse,
    ClusterLabelResult,
    ClusterMetadataCreate,
    ClusterMetadataListResponse,
    ClusterMetadataResponse,
    ClusterMetadataUpdate,
    ClusterStatsResponse,
    ProjectClusterResponse,
    ProjectDetectionRequest,
    ProjectDetectionResponse,
    ProjectStatus,
)
from priority_lens.schemas.cluster_metadata import (
    ProjectListResponse as ClusterProjectListResponse,
)
from priority_lens.schemas.inbox import (
    EmailSummary,
    InboxStats,
    PriorityContext,
    PriorityEmail,
    PriorityInboxResponse,
)
from priority_lens.schemas.oauth_token import (
    OAuthTokenCreate,
    OAuthTokenResponse,
    OAuthTokenStatus,
    OAuthTokenUpdate,
)
from priority_lens.schemas.org_user import (
    OrgUserCreate,
    OrgUserResponse,
    OrgUserUpdate,
)
from priority_lens.schemas.organization import (
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
)
from priority_lens.schemas.project import (
    ProjectCreate,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)
from priority_lens.schemas.sync import SyncStateResponse, SyncStateUpdate
from priority_lens.schemas.task import (
    TaskCreate,
    TaskDetailResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdate,
)
from priority_lens.schemas.watch import (
    WatchSubscriptionCreate,
    WatchSubscriptionResponse,
    WatchSubscriptionStatus,
    WatchSubscriptionUpdate,
)

__all__ = [
    "ClusterDimension",
    "ClusterLabelRequest",
    "ClusterLabelResponse",
    "ClusterLabelResult",
    "ClusterMetadataCreate",
    "ClusterMetadataListResponse",
    "ClusterMetadataResponse",
    "ClusterMetadataUpdate",
    "ClusterProjectListResponse",
    "ClusterStatsResponse",
    "ProjectClusterResponse",
    "ProjectDetectionRequest",
    "ProjectDetectionResponse",
    "ProjectStatus",
    "EmailSummary",
    "InboxStats",
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
    "PriorityContext",
    "PriorityEmail",
    "PriorityInboxResponse",
    "ProjectCreate",
    "ProjectDetailResponse",
    "ProjectListResponse",
    "ProjectResponse",
    "ProjectUpdate",
    "SyncStateResponse",
    "SyncStateUpdate",
    "TaskCreate",
    "TaskDetailResponse",
    "TaskListResponse",
    "TaskResponse",
    "TaskStatusUpdate",
    "TaskUpdate",
    "WatchSubscriptionCreate",
    "WatchSubscriptionResponse",
    "WatchSubscriptionStatus",
    "WatchSubscriptionUpdate",
]
