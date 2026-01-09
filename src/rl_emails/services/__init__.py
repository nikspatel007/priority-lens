"""Service layer for business logic orchestration."""

from rl_emails.services.auth_service import AuthService
from rl_emails.services.batch_processor import BatchProcessor, BatchResult
from rl_emails.services.entity_extraction import (
    ExtractionResult,
    PriorityContextBuilder,
    ProjectExtractor,
    TaskExtractor,
    extract_all_entities,
)
from rl_emails.services.progressive_sync import (
    PhaseConfig,
    ProgressiveSyncResult,
    ProgressiveSyncService,
    SyncPhase,
    SyncProgress,
)
from rl_emails.services.sync_service import SyncResult, SyncService

__all__ = [
    "AuthService",
    "BatchProcessor",
    "BatchResult",
    "ExtractionResult",
    "PhaseConfig",
    "PriorityContextBuilder",
    "ProgressiveSyncResult",
    "ProgressiveSyncService",
    "ProjectExtractor",
    "SyncPhase",
    "SyncProgress",
    "SyncResult",
    "SyncService",
    "TaskExtractor",
    "extract_all_entities",
]
