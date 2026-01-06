# Phase 3: Email Embeddings - Proposal

## What We're Building

A semantic embedding system that converts email content into dense vector representations, enabling similarity search, clustering, and content-based ranking.

## Why Embeddings Matter

### The Problem with Keyword Search
Current Phase 2 features use **exact patterns** (keywords, sender domains). This misses:
- "Can you review this?" vs "Please take a look" (same intent, different words)
- "Meeting tomorrow at 3pm" vs "Let's sync up Thursday afternoon" (similar meaning)
- Emails about the same project but using different terminology

### What Embeddings Enable

| Capability | Example |
|------------|---------|
| **Semantic Search** | Find emails about "budget concerns" even if they say "cost issues" |
| **Similar Email Clustering** | Group all emails about a project automatically |
| **Content-Based Priority** | "This email is similar to ones you replied to quickly" |
| **Anomaly Detection** | "This email is unlike anything from this sender" |
| **Smart Threading** | Group related conversations beyond reply chains |

## What We're Embedding

### Primary: Subject + Body Combined
```
[SUBJECT] Meeting rescheduled to Friday
[BODY] Hi team, due to the holiday, I'm moving our sync to Friday 2pm.
Please confirm availability. Thanks, Sarah
```

**Why combined?**
- Subject provides context/intent
- Body provides detail
- Together they capture full semantic meaning

### Embedding Model: OpenAI text-embedding-3-small

| Property | Value |
|----------|-------|
| Model | `text-embedding-3-small` |
| Dimensions | 1536 |
| Max tokens | 20000 |
| Cost | $0.02 / 1M tokens |
| Speed | ~100-200 emails/sec with batching |

**Why this model?**
- Best price/performance ratio
- 1536 dims sufficient for email similarity
- Handles long content well
- Widely used, stable API

### Cost Estimate

| Metric | Value |
|--------|-------|
| Emails | 22,618 |
| Avg tokens/email | ~300 (subject + body) |
| Total tokens | ~6.8M |
| Cost | ~$0.14 |

**Very affordable** - less than $0.15 for entire dataset.

## Storage Strategy

### Database: `email_embeddings` Table

```sql
CREATE TABLE email_embeddings (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id) UNIQUE,

    -- Embedding vector
    embedding VECTOR(1536),           -- pgvector type

    -- Metadata
    model TEXT DEFAULT 'text-embedding-3-small',
    token_count INTEGER,
    content_hash TEXT,                -- SHA256 of input text

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),

    -- Index for similarity search
    CONSTRAINT email_embeddings_email_id_key UNIQUE (email_id)
);

-- Create HNSW index for fast similarity search
CREATE INDEX ON email_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Local Backup: JSONL Files

```
backups/embeddings/
├── embeddings_20260106.jsonl      # Full export
├── embeddings_metadata.json       # Model info, stats
└── embeddings_batch_*.jsonl       # Incremental batches
```

**JSONL format:**
```json
{"email_id": 12345, "embedding": [0.123, -0.456, ...], "token_count": 287, "content_hash": "abc123..."}
```

**Why local backup?**
- Embeddings are expensive to recompute
- API costs add up over time
- Enables offline analysis
- Disaster recovery

## How We'll Use Embeddings

### 1. Find Similar Emails
```sql
-- Find emails similar to email #12345
SELECT e.subject, e.from_email,
       1 - (ee.embedding <=> target.embedding) as similarity
FROM email_embeddings ee
JOIN emails e ON e.id = ee.email_id
CROSS JOIN (
    SELECT embedding FROM email_embeddings WHERE email_id = 12345
) target
WHERE ee.email_id != 12345
ORDER BY ee.embedding <=> target.embedding
LIMIT 10;
```

### 2. Cluster by Topic
```sql
-- Find distinct email clusters (requires additional processing)
-- Embeddings enable K-means or DBSCAN clustering
```

### 3. Hybrid Ranking (Phase 2 + Phase 3)
```sql
-- Combine feature scores with semantic similarity
SELECT e.subject, e.from_email,
       ef.relationship_strength * 0.4 +
       ef.urgency_score * 0.3 +
       (1 - (ee.embedding <=> ref.embedding)) * 0.3 as hybrid_score
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
JOIN email_embeddings ee ON ee.email_id = e.id
CROSS JOIN (
    -- Reference: embeddings of emails you replied to quickly
    SELECT AVG(embedding) as embedding
    FROM email_embeddings ee2
    JOIN emails e2 ON e2.id = ee2.email_id
    WHERE e2.action = 'REPLIED'
      AND e2.response_time_seconds < 3600
) ref
ORDER BY hybrid_score DESC;
```

### 4. Semantic Search
```sql
-- Search for emails about "budget planning"
-- (Requires embedding the query first via API)
SELECT e.subject, e.from_email,
       1 - (ee.embedding <=> $1) as relevance
FROM email_embeddings ee
JOIN emails e ON e.id = ee.email_id
ORDER BY ee.embedding <=> $1
LIMIT 20;
```

### 5. Anomaly Detection
```sql
-- Find emails that are unusual for a sender
WITH sender_avg AS (
    SELECT e.from_email, AVG(ee.embedding) as avg_embedding
    FROM email_embeddings ee
    JOIN emails e ON e.id = ee.email_id
    GROUP BY e.from_email
    HAVING COUNT(*) > 5
)
SELECT e.subject, e.from_email,
       1 - (ee.embedding <=> sa.avg_embedding) as typicality
FROM email_embeddings ee
JOIN emails e ON e.id = ee.email_id
JOIN sender_avg sa ON sa.from_email = e.from_email
ORDER BY typicality ASC  -- Most unusual first
LIMIT 20;
```

## Implementation Plan

### Prerequisites
- [ ] Install pgvector extension in PostgreSQL
- [ ] Add OPENAI_API_KEY to .env
- [ ] Install litellm Python package (`uv pip install litellm`)
- [ ] Test API connectivity

### Why LiteLLM?

We use **LiteLLM** as our unified interface for all LLM/embedding calls:
- Single API for OpenAI, Anthropic, Cohere, local models, etc.
- Easy model switching without code changes
- Built-in retry logic and rate limiting
- Consistent interface across providers

```python
from litellm import embedding

# Works with any provider - just change model string
response = embedding(
    model="text-embedding-3-small",  # OpenAI
    # model="voyage/voyage-3",       # Voyage AI
    # model="cohere/embed-english-v3.0",  # Cohere
    input=["Your text here"]
)
embeddings = response.data[0]["embedding"]
```

### Script: `scripts/compute_embeddings.py`

```python
# Pseudocode structure
1. Load emails without embeddings
2. Prepare text: "[SUBJECT] {subject}\n[BODY] {body}"
3. Batch API calls via litellm.embedding() (100 emails per request)
4. Store in database + local JSONL backup
5. Track progress, handle rate limits
6. Verify coverage
```

### Batch Processing Strategy
- **Batch size**: 100 emails per API call
- **Rate limiting**: 3000 RPM for OpenAI
- **Checkpointing**: Save progress every 1000 emails
- **Resume capability**: Skip already-embedded emails

### Error Handling
- Retry on transient failures (429, 500, 503)
- Log and skip permanently failed emails
- Track failed email IDs for manual review

## Verification Queries

After running Phase 3:

```sql
-- Check coverage
SELECT
    (SELECT COUNT(*) FROM emails WHERE is_sent = FALSE) as total_received,
    (SELECT COUNT(*) FROM email_embeddings) as with_embeddings,
    ROUND(
        (SELECT COUNT(*) FROM email_embeddings) * 100.0 /
        (SELECT COUNT(*) FROM emails WHERE is_sent = FALSE), 1
    ) as coverage_pct;

-- Check embedding dimensions
SELECT
    vector_dims(embedding) as dims,
    COUNT(*) as count
FROM email_embeddings
GROUP BY dims;

-- Test similarity search
SELECT e.subject,
       1 - (ee.embedding <=> (SELECT embedding FROM email_embeddings LIMIT 1)) as sim
FROM email_embeddings ee
JOIN emails e ON e.id = ee.email_id
ORDER BY sim DESC
LIMIT 5;
```

## Dependencies

- ✅ Phase 1 complete (action labels)
- ✅ Phase 2 complete (ML features)
- [ ] pgvector extension installed
- [ ] OPENAI_API_KEY configured
- [ ] litellm Python package installed (`uv pip install litellm`)

## Files Created

1. `scripts/compute_embeddings.py` - Embedding generation script
2. `email_embeddings` table in database
3. `backups/embeddings/` - Local backup directory
4. Update `docs/v2/PHASE3_PROPOSAL.md` (this file)

## Cost Summary

| Item | Cost |
|------|------|
| Initial embedding (22,618 emails) | ~$0.14 |
| Incremental (new emails) | ~$0.00002/email |
| Storage (pgvector) | Free (included in PostgreSQL) |
| Local backup | Free (disk space) |

**Total initial cost: ~$0.14**

## What This Enables

**Immediate:**
- Semantic search across all emails
- Find similar emails to any email
- Cluster emails by topic

**Combined with Phase 2:**
- Hybrid ranking (features + semantics)
- "Emails similar to ones I replied to"
- Content-aware priority scoring

**Enables Phase 4 (LLM Integration):**

| Use Case | Embeddings Do | LLM Does |
|----------|---------------|----------|
| **RAG Search** | Find semantically relevant emails to query | Answer questions using retrieved context |
| **Smart Summarization** | Cluster related emails together | Summarize each cluster instead of individual emails |
| **Response Suggestions** | Find emails similar to ones you replied to | Generate draft responses based on your style |
| **Priority Explanation** | Find emails similar to high-priority ones | Explain *why* this email might need attention |
| **Thread Compression** | Identify redundant content in threads | Summarize key points, skip repetition |
| **Context Selection** | Select most relevant emails for limited context window | Generate insights from curated subset |
| **Deduplication** | Find near-duplicate forwards/FYIs | Avoid processing same content twice |

**Key insight**: Embeddings are cheap ($0.02/1M tokens). LLMs are expensive ($3-15/1M tokens). Use embeddings to pre-filter/select, then LLM only processes what matters.

---

**Ready to proceed?** Once approved, I'll:
1. Install pgvector extension
2. Create `compute_embeddings.py` script
3. Generate embeddings for all emails
4. Create local backup
5. Verify with similarity queries
6. Commit and push to rl-emails/main
