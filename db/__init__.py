"""SurrealDB integration for Email RL System.

This package provides SurrealDB-backed storage and querying for email data,
with efficient graph queries for communication patterns.

Usage:
    # Start SurrealDB:
    surreal start file:data/enron.db --user root --pass root

    # Import data:
    python -m db.import_data enron data/train.json data/val.json data/test.json

    # Use in training:
    from db.dataset import create_surreal_dataloaders
    train, val, test = create_surreal_dataloaders(database='enron')

Components:
    - schema.surql: Database schema with tables, edges, and functions
    - import_data.py: Import emails from JSON to SurrealDB
    - dataset.py: PyTorch Dataset backed by SurrealDB
"""

from .dataset import (
    SurrealEmailDataset,
    SurrealGraphQueries,
    create_surreal_dataloaders,
)

__all__ = [
    'SurrealEmailDataset',
    'SurrealGraphQueries',
    'create_surreal_dataloaders',
]
