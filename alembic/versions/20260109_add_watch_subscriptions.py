"""add_watch_subscriptions

Revision ID: 20260109_watch
Revises: fb71ba5949b0
Create Date: 2026-01-09

Add watch_subscriptions table for Gmail push notifications.
This table tracks Gmail API watch registrations that enable real-time
email notifications via Google Cloud Pub/Sub.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID


# revision identifiers, used by Alembic.
revision: str = "20260109_watch"
down_revision: str | Sequence[str] | None = "fb71ba5949b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create watch_subscriptions table."""
    op.create_table(
        "watch_subscriptions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        # Gmail watch details
        sa.Column("history_id", sa.String, nullable=True),
        sa.Column("expiration", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("topic_name", sa.String, nullable=True),
        sa.Column("label_ids", ARRAY(sa.String), nullable=True),
        # Status tracking
        sa.Column("status", sa.String, server_default="inactive", nullable=False),
        sa.Column("last_notification_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("notification_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("error_message", sa.String, nullable=True),
        # Timestamps
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Indexes
    op.create_index(
        "idx_watch_subscriptions_user_id",
        "watch_subscriptions",
        ["user_id"],
        unique=True,
    )
    op.create_index(
        "idx_watch_subscriptions_status",
        "watch_subscriptions",
        ["status"],
    )
    op.create_index(
        "idx_watch_subscriptions_expiration",
        "watch_subscriptions",
        ["expiration"],
    )


def downgrade() -> None:
    """Drop watch_subscriptions table."""
    op.drop_index("idx_watch_subscriptions_expiration", table_name="watch_subscriptions")
    op.drop_index("idx_watch_subscriptions_status", table_name="watch_subscriptions")
    op.drop_index("idx_watch_subscriptions_user_id", table_name="watch_subscriptions")
    op.drop_table("watch_subscriptions")
