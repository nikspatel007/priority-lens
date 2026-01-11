"""Celery application configuration.

This module configures Celery for background task processing.
Uses Redis as the broker and result backend.
"""

from __future__ import annotations

import os

from celery import Celery  # type: ignore[import-untyped]

# Redis URL from environment or default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "priority_lens",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["priority_lens.tasks.sync_tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Result expiration (24 hours)
    result_expires=86400,
    # Task tracking
    task_track_started=True,
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Concurrency - use threads for I/O bound tasks
    worker_concurrency=4,
    # Prefetch multiplier
    worker_prefetch_multiplier=1,
)
