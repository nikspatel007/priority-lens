"""Pipeline stages for email processing."""
from __future__ import annotations

from rl_emails.pipeline import (
    classify_ai_handleability,
    cluster_emails,
    compute_basic_features,
    compute_embeddings,
    compute_priority,
    enrich_emails_db,
    import_to_postgres,
    parse_mbox,
    populate_threads,
    populate_users,
    run_llm_classification,
)

__all__ = [
    "classify_ai_handleability",
    "cluster_emails",
    "compute_basic_features",
    "compute_embeddings",
    "compute_priority",
    "enrich_emails_db",
    "import_to_postgres",
    "parse_mbox",
    "populate_threads",
    "populate_users",
    "run_llm_classification",
]
