# Data Onboarding Guide

Complete guide to processing your Gmail export through the full ML pipeline.

## Prerequisites

### 1. Software Requirements

```bash
# Python 3.11+
python --version

# UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Docker (for PostgreSQL)
docker --version
```

### 2. API Keys

Create a `.env` file in the project root:

```bash
# Required for embeddings (Phase 3)
OPENAI_API_KEY=sk-...

# Required for LLM classification (Phase 4)
# Use either OpenAI OR Anthropic
OPENAI_API_KEY=sk-...        # GPT-5-mini (recommended)
ANTHROPIC_API_KEY=sk-ant-... # Claude alternative

# Your email address (for relationship detection)
YOUR_EMAIL=you@example.com

# Database URL (default works with docker-compose)
DB_URL=postgresql://postgres:postgres@localhost:5433/gmail_twoyrs
```

### 3. Gmail Export

1. Go to [Google Takeout](https://takeout.google.com/)
2. Select only "Mail"
3. Choose MBOX format
4. Download and extract the archive
5. Locate the `.mbox` file (usually `All mail Including Spam and Trash.mbox`)

---

## Quick Start (One Command)

```bash
# Full pipeline from MBOX to ML-ready
uv run python scripts/onboard_data.py /path/to/your.mbox

# With options
uv run python scripts/onboard_data.py /path/to/your.mbox \
    --workers 10 \
    --skip-llm          # Skip LLM classification (saves cost)
```

---

## Pipeline Stages

The pipeline processes your email in 7 stages:

| Stage | Script | Description | Time | Cost |
|-------|--------|-------------|------|------|
| 1 | `parse_mbox.py` | Parse MBOX → JSONL | ~2 min/10k | Free |
| 2 | `import_to_postgres.py` | Import to PostgreSQL | ~1 min/10k | Free |
| 3 | `populate_threads.py` | Build thread relationships | ~30 sec | Free |
| 4 | `compute_basic_features.py` | ML features (30 dimensions) | ~2 min/10k | Free |
| 5 | `compute_embeddings.py` | OpenAI embeddings | ~15 min/10k | ~$0.01/1k |
| 6 | `classify_ai_handleability.py` | Rule-based classification | ~1 min | Free |
| 7 | `run_llm_classification.py` | LLM classification | ~25 min/7k | ~$0.50/7k |

**Total for 22k emails**: ~45 minutes, ~$0.65

---

## Manual Step-by-Step

### Step 1: Setup Environment

```bash
# Clone and setup
cd rl-emails
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Start PostgreSQL
docker compose up -d postgres

# Or manually:
docker run -d \
  --name rl-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=gmail_twoyrs \
  -p 5433:5432 \
  pgvector/pgvector:pg16
```

### Step 2: Parse MBOX

```bash
# Set paths
export MBOX_PATH=/path/to/your.mbox
export PARSED_JSONL=./data/parsed_emails.jsonl

# Parse
uv run python scripts/parse_mbox.py

# Output: parsed_emails.jsonl (~500MB for 20k emails)
```

### Step 3: Import to Database

```bash
uv run python scripts/import_to_postgres.py

# Verify
uv run python -c "
import psycopg2
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5433/gmail_twoyrs')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM emails')
print(f'Imported: {cur.fetchone()[0]} emails')
"
```

### Step 4: Build Thread Relationships

```bash
uv run python scripts/populate_threads.py
```

### Step 5: Compute ML Features (Phase 2)

```bash
# Computes 30 features per email:
# - Relationship strength (11 dims)
# - Service detection (6 dims)
# - Temporal patterns (8 dims)
# - Content signals (5 dims)

uv run python scripts/compute_basic_features.py

# Verify
uv run python scripts/compute_basic_features.py --verify
```

### Step 6: Generate Embeddings (Phase 3)

```bash
# Requires OPENAI_API_KEY
# Uses text-embedding-3-small (1536 dimensions)

uv run python scripts/compute_embeddings.py --workers 10

# Check progress
uv run python scripts/compute_embeddings.py --status
```

### Step 7: Rule-Based Classification (Phase 0)

```bash
# Classifies emails into:
# - ai_full: Can auto-handle (newsletters, notifications)
# - ai_partial: AI can draft, human approves
# - human_required: Needs human decision
# - needs_llm: Ambiguous, needs LLM

uv run python scripts/classify_ai_handleability.py
```

### Step 8: LLM Classification (Phase 4)

```bash
# Only processes 'needs_llm' emails
# Uses GPT-5-mini by default

# Check how many need processing
uv run python scripts/run_llm_classification.py --status

# Run classification
uv run python scripts/run_llm_classification.py --all 10 gpt5

# Or use Claude
uv run python scripts/run_llm_classification.py --all 10 sonnet
```

---

## Verify Results

```bash
# Full status check
uv run python -c "
import psycopg2

conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5433/gmail_twoyrs')
cur = conn.cursor()

print('=== Pipeline Status ===')

cur.execute('SELECT COUNT(*) FROM emails WHERE is_sent = FALSE')
print(f'Total received emails: {cur.fetchone()[0]:,}')

cur.execute('SELECT COUNT(*) FROM email_features')
print(f'ML Features computed: {cur.fetchone()[0]:,}')

cur.execute('SELECT COUNT(*) FROM email_embeddings')
print(f'Embeddings generated: {cur.fetchone()[0]:,}')

cur.execute('SELECT predicted_handleability, COUNT(*) FROM email_ai_classification GROUP BY 1 ORDER BY 2 DESC')
print('\\nRule-based classification:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]:,}')

cur.execute('SELECT COUNT(*) FROM email_llm_classification')
print(f'\\nLLM classified: {cur.fetchone()[0]:,}')

cur.execute('SELECT action_type, COUNT(*) FROM email_llm_classification GROUP BY 1 ORDER BY 2 DESC LIMIT 5')
print('\\nTop action types:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]:,}')
"
```

---

## Output Tables

After the pipeline completes, you'll have:

### emails
Main email table with parsed content and user behavior labels.

```sql
SELECT id, from_email, subject, action, date_parsed
FROM emails LIMIT 5;
```

### email_features
30 ML features per email.

```sql
SELECT email_id, relationship_strength, is_service_email, urgency_score
FROM email_features LIMIT 5;
```

### email_embeddings
1536-dimension embeddings for semantic similarity.

```sql
SELECT email_id, array_length(embedding, 1) as dims
FROM email_embeddings LIMIT 5;
```

### email_ai_classification
Rule-based classification results.

```sql
SELECT predicted_handleability, COUNT(*)
FROM email_ai_classification
GROUP BY 1;
```

### email_llm_classification
LLM-powered classification with raw prompts/responses.

```sql
SELECT email_id, action_type, urgency, ai_can_handle, one_liner
FROM email_llm_classification LIMIT 5;
```

---

## Troubleshooting

### PostgreSQL Connection Failed

```bash
# Check if running
docker ps | grep postgres

# Start if not running
docker compose up -d postgres

# Check logs
docker logs rl-postgres
```

### OpenAI API Error

```bash
# Verify key is set
echo $OPENAI_API_KEY

# Test API
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Resume Failed Pipeline

```bash
# Check what's complete
uv run python scripts/run_llm_classification.py --status

# Resume from specific step
uv run python scripts/onboard_data.py /path/to/your.mbox --start-from 5
```

### Out of Memory

```bash
# Reduce batch size
uv run python scripts/compute_embeddings.py --batch-size 50 --workers 5
```

---

## Cost Estimation

| Emails | Embeddings | LLM Classification | Total |
|--------|------------|-------------------|-------|
| 10,000 | ~$0.10 | ~$0.25 | ~$0.35 |
| 25,000 | ~$0.25 | ~$0.60 | ~$0.85 |
| 50,000 | ~$0.50 | ~$1.20 | ~$1.70 |
| 100,000 | ~$1.00 | ~$2.40 | ~$3.40 |

*LLM classification only runs on ~30% of emails (needs_llm category)*

---

## Next Steps

After onboarding:

1. **Explore Data**: `uv run streamlit run apps/labeling_ui.py`
2. **Train Models**: See `docs/training.md`
3. **Build Agent**: See `docs/product_plan.md`

---

## Checkpoints

Create backups at key stages:

```bash
# Create checkpoint
uv run python scripts/checkpoint.py create --name after_phase4

# List checkpoints
uv run python scripts/checkpoint.py list

# Restore if needed
uv run python scripts/checkpoint.py restore after_phase4
```
