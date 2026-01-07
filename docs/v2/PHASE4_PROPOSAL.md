# Phase 4: LLM-Powered Learning Pipeline

## Goal

Build an efficient pipeline that uses LLM to extract learning signals from emails, optimized for:
- **Minimal LLM calls** (cost-efficient at scale)
- **Maximum learning signal** (RL system learns fast with few examples)
- **Scalability** (works for thousands of users)

## The Cost Problem

| Operation | Cost | Speed |
|-----------|------|-------|
| Embeddings (Phase 3) | $0.02/1M tokens | 25 emails/sec |
| LLM (Claude Haiku) | $0.25/1M input | 5-10 emails/sec |
| LLM (Claude Sonnet) | $3/1M input | 2-5 emails/sec |

**Key insight**: For 22,618 emails:
- Embeddings: ~$0.14 (done)
- Haiku on ALL: ~$17
- Sonnet on ALL: ~$200

We can't afford LLM on every email at scale. Solution: **use cheap operations (embeddings, features) to select what matters, then LLM only processes the important subset**.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 4 Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 4A       │    │ 4B       │    │ 4C       │    │ 4D       │  │
│  │ Cluster  │───▶│ Rank     │───▶│ LLM      │───▶│ RL       │  │
│  │ (free)   │    │ (free)   │    │ ($$$)    │    │ Training │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │        │
│       ▼               ▼               ▼               ▼        │
│  email_clusters  priority_score  llm_features   training_pairs │
│  (~50 topics)    (0-1 score)    (structured)   (REPLIED/IGN)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 4A: Multi-Dimensional Clustering

### Design Decision: Separate Clusterings (Approach 3)

We considered three approaches:
1. **Hierarchical**: People/Service → Relationship → Content
2. **Multi-View**: Combined feature vector (embedding + features)
3. **Separate Clusterings**: Independent clusters per dimension, then cross-analyze

**Chosen: Approach 3 (Separate Clusterings)** because:
- We can see what each dimension contributes independently
- We can weight dimensions later based on predictive power
- It's interpretable: "this person cluster" vs "this content cluster"
- Enables analysis: "For content cluster X, what predicts reply?"

### Why Multiple Dimensions Matter

Data analysis revealed clear dimension separation:

| Dimension | Signal | Evidence |
|-----------|--------|----------|
| **People** | 22x | HIGH relationship: 77.8% reply vs MINIMAL: 3.5% |
| **Service Type** | 2x | Transactional 0.61 importance vs Marketing 0.28 |
| **Content** | TBD | Semantic similarity (embeddings) |
| **Behavior** | TBD | Action patterns + response times |

**Problem with pure content clustering**: Embeddings group "Amazon order" with "Amazon marketing", mix high-relationship person with low-relationship person on same topic. Content similarity ≠ importance similarity.

### The Five Clustering Dimensions

```
┌─────────────────────────────────────────────────────────────────┐
│                  Multi-Dimensional Clustering                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │ PEOPLE   │  │ CONTENT  │  │ BEHAVIOR │  │ SERVICE  │  │ TEMPORAL │
│  │ Clusters │  │ Clusters │  │ Clusters │  │ Clusters │  │ Clusters │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
│       │             │             │             │             │
│       ▼             ▼             ▼             ▼             ▼
│  Who matters   What topics   How you act   Service types  When patterns
│  Relationship  Projects      Reply/Ignore  Trans/Notif    Day/Hour
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Cross-Dimensional Analysis
                    "For person cluster A + content cluster B,
                     what's the reply rate?"
```

### Cluster Types

#### 1. People Clusters
Group senders by relationship pattern, not just email address.

**Features**:
- relationship_strength
- user_replied_to_sender_rate
- avg_response_time_hours
- emails_from_sender_all
- sender_replies_to_you_rate

**Expected clusters**:
- Close collaborators (high interaction, fast response)
- Acquaintances (occasional, slower response)
- One-way senders (they email you, you rarely reply)
- New contacts (recent, pattern unknown)

#### 2. Content Clusters
Group by semantic similarity (what the email is about).

**Features**:
- Embedding vectors (reduced via UMAP to ~50 dims)

**Expected clusters**:
- Project-specific threads
- Meeting coordination
- Financial/orders
- News/updates
- Personal conversations

#### 3. Behavior Clusters
Group by how YOU responded (outcome patterns).

**Features**:
- action (REPLIED, ARCHIVED, IGNORED)
- response_time_seconds
- timing (immediate, same_day, delayed)

**Expected clusters**:
- Quick responders (replied < 1 hour)
- Delayed responders (replied > 24 hours)
- Archive-and-forget
- Ignored despite features

#### 4. Service Clusters
Group automated/service emails by type.

**Features**:
- service_type
- service_importance
- from_common_service_domain
- has_unsubscribe_link

**Expected clusters**:
- Transactional (orders, receipts) - important
- Notifications (alerts, updates) - sometimes important
- Newsletters (content) - rarely urgent
- Marketing (promotions) - low importance

#### 5. Temporal Clusters
Group by time patterns.

**Features**:
- hour_of_day
- day_of_week
- is_business_hours
- is_weekend

**Expected clusters**:
- Business hours senders
- After-hours/weekend
- Batch senders (same time daily)

### Output Tables

```sql
-- Main clustering results (one row per email per dimension)
CREATE TABLE email_clusters (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id),

    -- Cluster assignments per dimension
    people_cluster_id INTEGER,
    content_cluster_id INTEGER,
    behavior_cluster_id INTEGER,
    service_cluster_id INTEGER,
    temporal_cluster_id INTEGER,

    -- Confidence scores
    people_cluster_prob FLOAT,
    content_cluster_prob FLOAT,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(email_id)
);

-- Cluster metadata per dimension
CREATE TABLE cluster_metadata (
    id SERIAL PRIMARY KEY,
    dimension TEXT,              -- 'people', 'content', 'behavior', etc.
    cluster_id INTEGER,

    -- Size and stats
    size INTEGER,
    representative_email_id INTEGER,

    -- Auto-generated label
    auto_label TEXT,

    -- Behavioral stats for this cluster
    pct_replied FLOAT,
    avg_response_time_hours FLOAT,
    avg_relationship_strength FLOAT,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(dimension, cluster_id)
);

-- Cross-dimensional analysis cache
CREATE TABLE cluster_cross_stats (
    id SERIAL PRIMARY KEY,

    -- Dimension pair
    dim1 TEXT,
    cluster1_id INTEGER,
    dim2 TEXT,
    cluster2_id INTEGER,

    -- Stats for this intersection
    email_count INTEGER,
    pct_replied FLOAT,
    avg_response_time_hours FLOAT,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(dim1, cluster1_id, dim2, cluster2_id)
);
```

### Algorithm Choice

| Dimension | Algorithm | Why |
|-----------|-----------|-----|
| People | K-Means (k=10-20) | Clear relationship tiers |
| Content | HDBSCAN | Auto-discover topic count |
| Behavior | K-Means (k=5) | Known outcome categories |
| Service | Rule-based + K-Means | Service types known |
| Temporal | K-Means (k=5) | Time patterns are cyclic |

### Expected Results

| Dimension | Expected Clusters | Key Insight |
|-----------|-------------------|-------------|
| People | 10-20 | "These 5 people get 80% of your replies" |
| Content | 50-200 | "Project X emails vs newsletters vs orders" |
| Behavior | 5-10 | "Quick reply vs slow reply vs ignore" |
| Service | 5-10 | "Transactional vs marketing vs social" |
| Temporal | 5 | "Business hours vs evenings vs weekends" |

### Cross-Dimensional Questions

After clustering, we can answer:
- "For emails from people cluster 'close collaborators' about content cluster 'project X', what's the reply rate?"
- "Are 'notification' service emails more likely to be replied to during business hours?"
- "Which content clusters have the most 'quick reply' behavior?"

### Cost: FREE (local computation)

---

## Phase 4B: Hybrid Ranking

### What It Does
Combine Phase 2 features + Phase 3 embeddings to score every email's importance.

### Ranking Formula

```python
priority_score = (
    # Phase 2 features (40%)
    relationship_strength * 0.15 +
    urgency_score * 0.15 +
    (1 - is_service_email) * 0.10 +

    # Behavioral signal (30%)
    replied_similarity * 0.20 +      # Similar to emails you replied to
    response_time_factor * 0.10 +    # You replied quickly to similar

    # Novelty (30%)
    cluster_novelty * 0.15 +         # Unusual for this cluster
    sender_novelty * 0.15            # Unusual for this sender
)
```

### Key Metrics

**replied_similarity**: Cosine similarity to centroid of REPLIED emails
```sql
-- Compute centroid of emails you replied to
SELECT AVG(embedding) as replied_centroid
FROM email_embeddings ee
JOIN emails e ON e.id = ee.email_id
WHERE e.action = 'REPLIED';

-- Score each email by similarity to this centroid
SELECT email_id, 1 - (embedding <=> replied_centroid) as replied_similarity
FROM email_embeddings;
```

**cluster_novelty**: How unusual is this email for its cluster?
```sql
-- Emails far from their cluster centroid are novel
SELECT email_id,
       1 - (embedding <=> cluster_centroid) as typicality,
       1 - typicality as novelty
FROM email_embeddings ee
JOIN cluster_metadata cm ON ...
```

### Output: `email_priority` Table

```sql
CREATE TABLE email_priority (
    email_id INTEGER PRIMARY KEY REFERENCES emails(id),

    -- Component scores
    feature_score FLOAT,          -- From Phase 2 features
    replied_similarity FLOAT,     -- Similar to replied emails
    cluster_novelty FLOAT,        -- Unusual for cluster
    sender_novelty FLOAT,         -- Unusual for sender

    -- Final score
    priority_score FLOAT,         -- 0-1 combined score
    priority_rank INTEGER,        -- 1 = highest priority

    -- Flags for LLM processing
    needs_llm_analysis BOOLEAN,   -- Should LLM look at this?
    llm_reason TEXT,              -- Why? (high priority, novel, etc.)

    created_at TIMESTAMP DEFAULT NOW()
);
```

### LLM Selection Strategy

Not all emails need LLM. Select based on:

| Category | Criteria | LLM Action |
|----------|----------|------------|
| **High Priority** | priority_score > 0.7 | Full analysis |
| **Cluster Representatives** | is_representative = true | Summarize cluster |
| **Novel/Anomalous** | cluster_novelty > 0.8 | Investigate |
| **Replied (training data)** | action = 'REPLIED' | Extract what made it important |
| **Ignored (training data)** | action = 'IGNORED' AND priority_score > 0.5 | Why was this ignored? |

**Expected LLM volume**: ~2,000-3,000 emails (10-15% of dataset)

### Cost: FREE (local computation)

---

## Phase 4C: LLM Feature Extraction

### What It Does
For selected emails, use LLM to extract structured learning signals.

### LLM Tasks

#### 1. Importance Analysis
```
Given this email, explain:
1. Why might this need attention? (or why not?)
2. What action does it require? (reply, task, FYI, none)
3. What's the deadline/urgency? (immediate, this week, none)
4. Key entities: people, projects, amounts mentioned
```

#### 2. Cluster Summarization
```
Given these 5 representative emails from a cluster:
1. What topic/theme unites them?
2. Suggested label (2-3 words)
3. Typical required action for this type
4. Importance pattern (always important, rarely, depends on...)
```

#### 3. Training Signal Extraction
```
This email was REPLIED to within 2 hours.
1. What signals indicated this needed a quick reply?
2. What keywords/patterns suggest urgency?
3. What about the sender/content made this important?

This email was IGNORED despite high feature scores.
1. Why might this have been ignored?
2. What signals indicate low importance despite surface urgency?
```

### Output: `llm_features` Table

```sql
CREATE TABLE llm_features (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id),

    -- Structured extraction
    llm_importance_score FLOAT,      -- 0-1, LLM's assessment
    llm_action_type TEXT,            -- 'reply', 'task', 'fyi', 'ignore'
    llm_urgency TEXT,                -- 'immediate', 'this_week', 'none'
    llm_explanation TEXT,            -- Why this importance?

    -- Entities
    mentioned_people TEXT[],
    mentioned_projects TEXT[],
    mentioned_deadlines TIMESTAMP[],
    mentioned_amounts TEXT[],

    -- Training signals
    importance_signals TEXT[],       -- What made this important
    ignore_signals TEXT[],           -- What suggests low importance

    -- Metadata
    model TEXT,                      -- 'claude-3-haiku', etc.
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd FLOAT,

    created_at TIMESTAMP DEFAULT NOW()
);
```

### Cost Optimization

| Strategy | Savings |
|----------|---------|
| Only process selected emails (10-15%) | 85-90% |
| Use Haiku for most, Sonnet for complex | 50% |
| Batch similar emails in one prompt | 30% |
| Cache cluster summaries | 20% |

**Estimated cost for 22K emails**: ~$2-5 (vs $17-200 naive approach)

---

## Phase 4D: RL Training Data

### What It Does
Convert LLM insights + user behavior into preference pairs for RL training.

### Preference Pair Format

```json
{
  "chosen": {
    "email_id": 1234,
    "action": "REPLIED",
    "response_time": 1800,
    "features": {...},
    "llm_importance": 0.85,
    "llm_signals": ["deadline mentioned", "direct question"]
  },
  "rejected": {
    "email_id": 5678,
    "action": "IGNORED",
    "features": {...},
    "llm_importance": 0.3,
    "llm_signals": ["newsletter", "no action required"]
  },
  "context": "Both from same sender, similar time period"
}
```

### Pair Selection Strategy

**High-quality pairs** (clear signal):
- REPLIED quickly vs IGNORED from same sender
- REPLIED vs IGNORED in same cluster
- High LLM importance + REPLIED vs Low LLM importance + IGNORED

**Hard pairs** (model needs to learn nuance):
- REPLIED slowly vs IGNORED (borderline cases)
- High feature score + IGNORED (false positives to avoid)
- Low feature score + REPLIED (hidden importance)

### Output: `training_pairs` Table

```sql
CREATE TABLE training_pairs (
    id SERIAL PRIMARY KEY,

    chosen_email_id INTEGER REFERENCES emails(id),
    rejected_email_id INTEGER REFERENCES emails(id),

    -- Pair metadata
    pair_type TEXT,              -- 'same_sender', 'same_cluster', 'temporal'
    difficulty TEXT,             -- 'easy', 'medium', 'hard'

    -- Signals
    chosen_signals JSONB,
    rejected_signals JSONB,

    -- Quality score (for curriculum learning)
    pair_quality FLOAT,

    created_at TIMESTAMP DEFAULT NOW()
);
```

### Expected Output
- ~5,000-10,000 high-quality pairs from 22K emails
- Stratified by: sender type, cluster, difficulty
- Ready for DPO/RLHF training

---

## Implementation Plan

### Phase 4A: Clustering
```bash
python scripts/cluster_emails.py
```
- Input: email_embeddings
- Output: email_clusters, cluster_metadata
- Time: ~5 minutes (local)
- Cost: FREE

### Phase 4B: Ranking
```bash
python scripts/compute_priority.py
```
- Input: email_features, email_embeddings, email_clusters
- Output: email_priority
- Time: ~10 minutes (local)
- Cost: FREE

### Phase 4C: LLM Extraction
```bash
python scripts/extract_llm_features.py --budget 5.00
```
- Input: email_priority (selected subset)
- Output: llm_features
- Time: ~30-60 minutes (API bound)
- Cost: ~$2-5

### Phase 4D: Training Pairs
```bash
python scripts/build_training_pairs.py
```
- Input: emails, llm_features, email_priority
- Output: training_pairs
- Time: ~5 minutes (local)
- Cost: FREE

---

## Scaling to Thousands of Users

### Per-User Cost Model

| Component | First User | Additional Users |
|-----------|------------|------------------|
| Embeddings | $0.14 | $0.14 per user |
| Clustering | FREE | FREE |
| Ranking | FREE | FREE |
| LLM (10%) | ~$3 | ~$3 per user |
| **Total** | ~$3.50 | ~$3.50 per user |

### Amortized Learning

Once we have training data from early users:
- Train base model on aggregated preferences
- New users benefit from existing model
- LLM calls decrease as model improves
- Eventually: model predictions replace most LLM calls

### Architecture for Scale

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-User Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Emails → Embeddings → Clustering → Ranking            │
│       │                          │           │              │
│       │                          ▼           ▼              │
│       │                    ┌─────────────────────┐          │
│       │                    │   Shared Base Model │          │
│       │                    │   (trained on all   │          │
│       │                    │    user data)       │          │
│       │                    └─────────────────────┘          │
│       │                              │                      │
│       │                              ▼                      │
│       │                    Model predicts importance        │
│       │                              │                      │
│       │                              ▼                      │
│       │              Low confidence? ──────▶ LLM call       │
│       │                    │                    │           │
│       │                    ▼                    ▼           │
│       └──────────▶  Final priority score  ◀────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

### Phase 4A (Clustering)
- [ ] 50-200 meaningful clusters discovered
- [ ] >90% of emails assigned to clusters
- [ ] Cluster labels are interpretable

### Phase 4B (Ranking)
- [ ] Priority scores correlate with actual replies (AUC > 0.7)
- [ ] Top 10% by score contains >50% of replied emails
- [ ] Novel emails surfaced (not just high-feature emails)

### Phase 4C (LLM)
- [ ] <$5 total cost for full dataset
- [ ] Structured extraction success rate >95%
- [ ] LLM importance correlates with user behavior

### Phase 4D (Training)
- [ ] >5,000 high-quality preference pairs
- [ ] Pairs span all clusters/sender types
- [ ] Hard pairs included for nuanced learning

---

## Dependencies

- ✅ Phase 1 complete (action labels)
- ✅ Phase 2 complete (ML features)
- ✅ Phase 3 complete (embeddings)
- [ ] HDBSCAN installed (`uv pip install hdbscan`)
- [ ] LiteLLM configured for Claude

---

## Files to Create

1. `scripts/cluster_emails.py` - Phase 4A
2. `scripts/compute_priority.py` - Phase 4B
3. `scripts/extract_llm_features.py` - Phase 4C
4. `scripts/build_training_pairs.py` - Phase 4D
5. `docs/v2/PHASE4_PROPOSAL.md` - This document

---

**Ready to proceed?** Recommend starting with Phase 4A (clustering) - it's free and provides foundation for everything else.
