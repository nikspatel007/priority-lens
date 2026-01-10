"""Pipeline module for email ML processing.

This module provides the email processing pipeline with 11 stages:
1. Parse MBOX to JSONL
2. Import to PostgreSQL
3. Build thread relationships
4. Compute action labels
5. Compute ML features
6. Generate embeddings
7. Rule-based classification
8. Populate user profiles
9. Multi-dimensional clustering
10. Hybrid priority ranking
11. LLM classification
"""

from __future__ import annotations

from priority_lens.pipeline.orchestrator import (
    PipelineOptions,
    PipelineOrchestrator,
    PipelineResult,
)
from priority_lens.pipeline.stages.base import StageResult
from priority_lens.pipeline.status import (
    PipelineStatus,
    check_postgres,
    format_status,
    get_status,
)

__all__ = [
    "PipelineOptions",
    "PipelineOrchestrator",
    "PipelineResult",
    "PipelineStatus",
    "StageResult",
    "check_postgres",
    "format_status",
    "get_status",
]
