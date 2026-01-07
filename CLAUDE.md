# rl-emails Project Guide

## Overview

Email ML pipeline for analyzing Gmail exports and predicting email priority/actions.

**Status**: Production-ready onboarding pipeline

## What Works (Production Ready)

### Onboarding Pipeline (11 stages)

```
scripts/
├── onboard_data.py              # Main orchestrator
├── parse_mbox.py                # Stage 1: Parse MBOX → JSONL
├── import_to_postgres.py        # Stage 2: Import to PostgreSQL
├── populate_threads.py          # Stage 3: Build thread relationships
├── enrich_emails_db.py          # Stage 4: Compute action labels (Phase 1)
├── compute_basic_features.py    # Stage 5: Compute ML features (Phase 2)
├── compute_embeddings.py        # Stage 6: Generate embeddings (Phase 3)
├── classify_ai_handleability.py # Stage 7: Rule-based classification (Phase 0)
├── populate_users.py            # Stage 8: User profiles (Phase 4A)
├── cluster_emails.py            # Stage 9: Multi-dimensional clustering (Phase 4B)
├── compute_priority.py          # Stage 10: Hybrid priority ranking (Phase 4C)
├── run_llm_classification.py    # Stage 11: LLM classification (Phase 4D)
├── checkpoint.py                # Utility: Checkpoint/restore
├── query_db.py                  # Utility: Database queries
└── validate_data.py             # Utility: Data validation
```

### Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment
cp .env.example .env
# Edit .env with your DATABASE_URL, MBOX_PATH, YOUR_EMAIL, OPENAI_API_KEY

# 3. Start PostgreSQL
docker compose up -d

# 4. Run pipeline
uv run python scripts/onboard_data.py

# 5. Check status
uv run python scripts/onboard_data.py --status

# 6. Validate data
uv run python scripts/validate_data.py
```

### Environment Variables

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/gmail_test_30d
MBOX_PATH=/path/to/your/gmail.mbox
YOUR_EMAIL=your_email@example.com
OPENAI_API_KEY=sk-...
```

## Database Schema

Key tables populated by the pipeline:

| Table | Purpose |
|-------|---------|
| `emails` | Raw email data |
| `threads` | Conversation groupings |
| `email_features` | ML features (relationship_strength, urgency, etc.) |
| `email_embeddings` | OpenAI embeddings (1536 dimensions) |
| `email_ai_classification` | Rule-based handleability classification |
| `email_llm_classification` | LLM-based classification (gpt-5-mini) |
| `users` | User profiles with reply rates |
| `email_clusters` | Multi-dimensional clustering |
| `email_priority` | Hybrid priority scores |

## Key Metrics

After running the pipeline, validate with:

```bash
# Check relationship strength correlation
uv run python scripts/validate_data.py

# Expected: Strong relationships → higher reply rates
# Strong (0.7+):  ~77% reply rate
# Medium (0.5+):  ~58% reply rate
# Weak (0.3+):    ~1% reply rate
```

## Development

### Run Tests

```bash
uv run pytest tests/ -v
```

### Check Pipeline Status

```bash
uv run python scripts/onboard_data.py --status
```

### Create Checkpoint

```bash
uv run python scripts/checkpoint.py create --name my_checkpoint
```

### Restore Checkpoint

```bash
uv run python scripts/checkpoint.py restore --name my_checkpoint
```

## Production Plan

See `ralph-wiggum.md` for the iteration plan to achieve:
- 100% type coverage (mypy strict)
- 100% test coverage
- Makefile-based workflow
- Production-ready codebase

## File Structure

```
rl-emails/
├── scripts/           # Production pipeline (15 files)
├── tests/             # Test suite
├── alembic/           # Database migrations
├── data/              # Email data and outputs
├── archive/           # Legacy/experimental code (not used)
├── .env               # Environment configuration
├── pyproject.toml     # Project dependencies
├── ralph-wiggum.md    # Production iteration plan
└── CLAUDE.md          # This file
```

## NOT Production Ready

The following directories contain experimental/legacy code and are NOT used by the pipeline:

- `src/` - Experimental feature modules and training code
- `apps/` - Legacy UI code
- `archive/` - Previously archived scripts

These should be cleaned up per the `ralph-wiggum.md` plan.
