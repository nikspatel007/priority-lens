"""Stage 12: Entity extraction.

Extracts structured entities from pipeline data:
- Tasks from LLM and AI classifications
- Projects from cluster metadata
- Priority contexts from email features
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sqlalchemy import create_engine

from priority_lens.pipeline.stages.base import StageResult
from priority_lens.services.entity_extraction import extract_all_entities

if TYPE_CHECKING:
    from priority_lens.core.config import Config


def run(config: Config, **kwargs: object) -> StageResult:
    """Run entity extraction stage.

    Args:
        config: Pipeline configuration.
        **kwargs: Additional options (user_id for multi-tenant).

    Returns:
        StageResult with extraction statistics.
    """
    start_time = time.time()

    # Get user_id from config if in multi-tenant mode
    user_id = config.user_id if config.is_multi_tenant else None

    # Create SQLAlchemy engine and connection
    engine = create_engine(config.sync_database_url)

    with engine.connect() as conn:
        # Run extraction
        result = extract_all_entities(conn, user_id)

        # Commit the transaction
        conn.commit()

    elapsed = time.time() - start_time

    # Build message
    messages = []
    messages.append(f"Tasks: {result.tasks_created} created, {result.tasks_updated} updated")
    messages.append(
        f"Projects: {result.projects_created} created, {result.projects_updated} updated"
    )
    messages.append(
        f"Contexts: {result.priority_contexts_created} created, "
        f"{result.priority_contexts_updated} updated"
    )

    if result.errors:
        messages.append(f"Errors: {len(result.errors)}")

    return StageResult(
        success=len(result.errors) == 0,
        message="; ".join(messages),
        duration_seconds=elapsed,
        records_processed=result.total_created + result.total_updated,
    )
