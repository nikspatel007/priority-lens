"""add_entity_extraction_columns

Revision ID: fb71ba5949b0
Revises: 20260107144608
Create Date: 2026-01-08

Add user_id columns to projects and tasks tables for multi-tenant support.
Add additional columns for richer entity extraction.
Enhance priority_contexts table schema.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = "fb71ba5949b0"
down_revision: Union[str, Sequence[str], None] = "20260107144608"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_column_if_not_exists(table: str, column: str, col_type: str) -> None:
    """Add a column only if it doesn't already exist."""
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = '{table}' AND column_name = '{column}'
            ) THEN
                ALTER TABLE {table} ADD COLUMN {column} {col_type};
            END IF;
        END $$;
        """
    )


def _create_index_if_not_exists(idx_name: str, table: str, column: str) -> None:
    """Create an index only if it doesn't already exist."""
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = '{idx_name}'
            ) THEN
                CREATE INDEX {idx_name} ON {table} ({column});
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    """Add entity extraction columns (idempotent)."""

    # =================================================================
    # ENHANCE PROJECTS TABLE
    # =================================================================

    # Add user_id for multi-tenant support
    _add_column_if_not_exists(
        "projects", "user_id", "UUID REFERENCES org_users(id) ON DELETE CASCADE"
    )
    _add_column_if_not_exists("projects", "project_type", "TEXT")
    _add_column_if_not_exists("projects", "owner_email", "TEXT")
    _add_column_if_not_exists("projects", "participants", "TEXT[]")
    _add_column_if_not_exists("projects", "start_date", "TIMESTAMPTZ")
    _add_column_if_not_exists("projects", "due_date", "TIMESTAMPTZ")
    _add_column_if_not_exists("projects", "completed_at", "TIMESTAMPTZ")
    _add_column_if_not_exists("projects", "email_count", "INTEGER DEFAULT 0")
    _add_column_if_not_exists("projects", "last_activity", "TIMESTAMPTZ")
    _add_column_if_not_exists("projects", "detected_from", "TEXT")
    _add_column_if_not_exists("projects", "cluster_id", "INTEGER")
    _add_column_if_not_exists("projects", "confidence", "FLOAT")

    _create_index_if_not_exists("idx_projects_user_id", "projects", "user_id")

    # =================================================================
    # ENHANCE TASKS TABLE
    # =================================================================

    _add_column_if_not_exists(
        "tasks", "user_id", "UUID REFERENCES org_users(id) ON DELETE CASCADE"
    )
    _add_column_if_not_exists(
        "tasks", "project_id", "INTEGER REFERENCES projects(id) ON DELETE SET NULL"
    )
    _add_column_if_not_exists("tasks", "deadline_type", "TEXT")
    _add_column_if_not_exists("tasks", "assigned_to", "TEXT")
    _add_column_if_not_exists("tasks", "assigned_by", "TEXT")
    _add_column_if_not_exists("tasks", "is_assigned_to_user", "BOOLEAN DEFAULT false")
    _add_column_if_not_exists("tasks", "assignment_confidence", "FLOAT")
    _add_column_if_not_exists("tasks", "extraction_method", "TEXT")
    _add_column_if_not_exists("tasks", "status", "TEXT DEFAULT 'pending'")
    _add_column_if_not_exists("tasks", "completed_at", "TIMESTAMPTZ")

    _create_index_if_not_exists("idx_tasks_user_id", "tasks", "user_id")
    _create_index_if_not_exists("idx_tasks_project_id", "tasks", "project_id")

    # =================================================================
    # ENHANCE PRIORITY_CONTEXTS TABLE
    # =================================================================

    _add_column_if_not_exists(
        "priority_contexts",
        "user_id",
        "UUID REFERENCES org_users(id) ON DELETE CASCADE",
    )

    # Sender context columns
    _add_column_if_not_exists("priority_contexts", "sender_email", "TEXT")
    _add_column_if_not_exists("priority_contexts", "sender_frequency", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "sender_importance", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "sender_reply_rate", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "sender_org_level", "INTEGER")
    _add_column_if_not_exists(
        "priority_contexts", "sender_relationship_strength", "FLOAT"
    )

    # Thread context columns
    _add_column_if_not_exists("priority_contexts", "thread_id", "TEXT")
    _add_column_if_not_exists("priority_contexts", "is_reply", "BOOLEAN")
    _add_column_if_not_exists("priority_contexts", "thread_length", "INTEGER")
    _add_column_if_not_exists("priority_contexts", "thread_depth", "INTEGER")
    _add_column_if_not_exists("priority_contexts", "user_already_replied", "BOOLEAN")

    # Temporal context columns
    _add_column_if_not_exists("priority_contexts", "email_timestamp", "TIMESTAMPTZ")
    _add_column_if_not_exists("priority_contexts", "is_business_hours", "BOOLEAN")
    _add_column_if_not_exists("priority_contexts", "age_hours", "FLOAT")

    # Component scores
    _add_column_if_not_exists("priority_contexts", "people_score", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "project_score", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "topic_score", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "task_score", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "temporal_score", "FLOAT")
    _add_column_if_not_exists("priority_contexts", "relationship_score", "FLOAT")

    # Overall priority and timestamp
    _add_column_if_not_exists("priority_contexts", "overall_priority", "FLOAT")
    _add_column_if_not_exists(
        "priority_contexts", "computed_at", "TIMESTAMPTZ DEFAULT now()"
    )

    _create_index_if_not_exists(
        "idx_priority_contexts_user_id", "priority_contexts", "user_id"
    )
    _create_index_if_not_exists(
        "idx_priority_contexts_priority", "priority_contexts", "overall_priority"
    )


def downgrade() -> None:
    """Remove entity extraction columns."""

    # Priority contexts
    op.execute("DROP INDEX IF EXISTS idx_priority_contexts_priority")
    op.execute("DROP INDEX IF EXISTS idx_priority_contexts_user_id")
    op.drop_column("priority_contexts", "computed_at")
    op.drop_column("priority_contexts", "overall_priority")
    op.drop_column("priority_contexts", "relationship_score")
    op.drop_column("priority_contexts", "temporal_score")
    op.drop_column("priority_contexts", "task_score")
    op.drop_column("priority_contexts", "topic_score")
    op.drop_column("priority_contexts", "project_score")
    op.drop_column("priority_contexts", "people_score")
    op.drop_column("priority_contexts", "age_hours")
    op.drop_column("priority_contexts", "is_business_hours")
    op.drop_column("priority_contexts", "email_timestamp")
    op.drop_column("priority_contexts", "user_already_replied")
    op.drop_column("priority_contexts", "thread_depth")
    op.drop_column("priority_contexts", "thread_length")
    op.drop_column("priority_contexts", "is_reply")
    op.drop_column("priority_contexts", "thread_id")
    op.drop_column("priority_contexts", "sender_relationship_strength")
    op.drop_column("priority_contexts", "sender_org_level")
    op.drop_column("priority_contexts", "sender_reply_rate")
    op.drop_column("priority_contexts", "sender_importance")
    op.drop_column("priority_contexts", "sender_frequency")
    op.drop_column("priority_contexts", "sender_email")
    op.drop_column("priority_contexts", "user_id")

    # Tasks
    op.execute("DROP INDEX IF EXISTS idx_tasks_project_id")
    op.execute("DROP INDEX IF EXISTS idx_tasks_user_id")
    op.drop_column("tasks", "completed_at")
    op.drop_column("tasks", "status")
    op.drop_column("tasks", "extraction_method")
    op.drop_column("tasks", "assignment_confidence")
    op.drop_column("tasks", "is_assigned_to_user")
    op.drop_column("tasks", "assigned_by")
    op.drop_column("tasks", "assigned_to")
    op.drop_column("tasks", "deadline_type")
    op.drop_column("tasks", "project_id")
    op.drop_column("tasks", "user_id")

    # Projects
    op.execute("DROP INDEX IF EXISTS idx_projects_user_id")
    op.drop_column("projects", "confidence")
    op.drop_column("projects", "cluster_id")
    op.drop_column("projects", "detected_from")
    op.drop_column("projects", "last_activity")
    op.drop_column("projects", "email_count")
    op.drop_column("projects", "completed_at")
    op.drop_column("projects", "due_date")
    op.drop_column("projects", "start_date")
    op.drop_column("projects", "participants")
    op.drop_column("projects", "owner_email")
    op.drop_column("projects", "project_type")
    op.drop_column("projects", "user_id")
