"""add action column to human_task_labels

Revision ID: a1b2c3d4e5f6
Revises: 92f0657c25ef
Create Date: 2026-01-05

Adds 'action' column to human_task_labels table.
Action represents what the user will DO with the email:
- delete, archive, reply_now, reply_later, forward, create_task, snooze

This is separate from triage_category (categorization) - action is the intended behavior.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '92f0657c25ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add action column to human_task_labels."""
    op.add_column('human_task_labels', sa.Column('action', sa.Text, nullable=True))


def downgrade() -> None:
    """Remove action column from human_task_labels."""
    op.drop_column('human_task_labels', 'action')
