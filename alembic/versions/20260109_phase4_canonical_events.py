"""phase4_canonical_events

Revision ID: 20260109_phase4_events
Revises: 20260109_temporal
Create Date: 2026-01-09

Add canonical event schema for Voice AI + SDUI integration.
Tables: conversation_threads, sessions, events

This enables:
- Append-only event log for all agent interactions
- Replayable conversation history
- Multi-tenant scoping via org_id

Note: Using 'conversation_threads' to avoid conflict with existing 'threads'
table which is used for email threads.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision: str = "20260109_phase4_events"
down_revision: Union[str, Sequence[str], None] = "20260109_temporal"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create canonical event tables for Voice AI + SDUI."""

    # =================================================================
    # CONVERSATION_THREADS TABLE - Voice/text conversation threads
    # =================================================================
    op.create_table(
        "conversation_threads",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_conv_threads_org_id", "conversation_threads", ["org_id"])
    op.create_index("idx_conv_threads_user_id", "conversation_threads", ["user_id"])
    op.create_index(
        "idx_conv_threads_user_created",
        "conversation_threads",
        ["user_id", "created_at"],
    )

    # =================================================================
    # SESSIONS TABLE - Voice/text sessions within threads
    # =================================================================
    op.create_table(
        "sessions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "thread_id",
            UUID(as_uuid=True),
            sa.ForeignKey("conversation_threads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("mode", sa.Text, server_default="text"),  # text | voice
        sa.Column("status", sa.Text, server_default="active"),  # active | ended
        sa.Column("livekit_room", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column(
            "started_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("idx_sessions_thread_id", "sessions", ["thread_id"])
    op.create_index("idx_sessions_org_id", "sessions", ["org_id"])
    op.create_index("idx_sessions_status", "sessions", ["status"])

    # =================================================================
    # EVENTS TABLE - Append-only canonical event log
    # =================================================================
    op.create_table(
        "events",
        sa.Column(
            "event_id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "thread_id",
            UUID(as_uuid=True),
            sa.ForeignKey("conversation_threads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer, nullable=False),  # Monotonic per thread
        sa.Column("ts", sa.BigInteger, nullable=False),  # Epoch ms
        sa.Column("actor", sa.Text, nullable=False),  # user | agent | tool | system
        sa.Column("type", sa.Text, nullable=False),  # Event type
        sa.Column("payload", JSONB, server_default="{}"),
        sa.Column("correlation_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # Unique constraint on thread_id + seq for monotonic ordering
    op.create_index(
        "idx_events_thread_seq",
        "events",
        ["thread_id", "seq"],
        unique=True,
    )
    op.create_index("idx_events_thread_id", "events", ["thread_id"])
    op.create_index("idx_events_org_id", "events", ["org_id"])
    op.create_index("idx_events_type", "events", ["type"])
    op.create_index("idx_events_correlation_id", "events", ["correlation_id"])
    op.create_index("idx_events_session_id", "events", ["session_id"])
    op.create_index(
        "idx_events_thread_ts",
        "events",
        ["thread_id", "ts"],
    )


def downgrade() -> None:
    """Remove canonical event tables."""
    op.drop_table("events")
    op.drop_table("sessions")
    op.drop_table("conversation_threads")
