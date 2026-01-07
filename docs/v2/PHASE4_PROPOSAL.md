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

## Phase 4B.5: Pre-LLM AI Handleability Classification

### Why Pre-Classify Before LLM?

LLM extraction is expensive. Before spending on LLM, we can use rule-based classification to:
1. **Skip emails that don't need LLM** (newsletters, auto-file candidates)
2. **Identify emails that definitely need human** (approvals, high-relationship)
3. **Focus LLM budget on ambiguous cases** where we genuinely need understanding

### Rule-Based AI Handleability Results

Analysis of 22,618 emails shows clear segmentation:

| Category | Count | % of Emails | Reply Rate | What This Means |
|----------|-------|-------------|------------|-----------------|
| **ai_full** | 6,983 | 30.9% | 0.5% | AI could fully handle (file, skip) |
| **ai_partial** | 3,883 | 17.2% | 9.5% | AI could help (draft, schedule) |
| **human_required** | 1,674 | 7.4% | 20.1% | Needs human (decision, personal) |
| **needs_llm** | 10,078 | 44.6% | 3.3% | Need LLM to understand |

### Classification Rules

```python
def predict_ai_handleability(email, features):
    """
    Rule-based prediction of what AI could do with this email.
    Runs BEFORE LLM to reduce LLM volume.
    """

    # ═══════════════════════════════════════════════════════════════
    # AI_FULL: Can be completely handled by AI without human
    # ═══════════════════════════════════════════════════════════════

    # Newsletters, marketing, notifications - just file them
    if features.is_service_email and features.service_type in ['marketing', 'newsletter', 'notification']:
        return 'ai_full', {'action': 'file', 'folder': 'Promotions' if features.service_type == 'marketing' else 'Updates'}

    # Order confirmations, receipts - file to reference
    if features.is_service_email and features.service_type == 'transactional':
        if any(word in email.body.lower() for word in ['order', 'receipt', 'confirmation', 'shipped']):
            return 'ai_full', {'action': 'file', 'folder': 'Orders', 'track_delivery': 'shipped' in email.body.lower()}

    # ═══════════════════════════════════════════════════════════════
    # AI_PARTIAL: AI can prepare, human finishes
    # ═══════════════════════════════════════════════════════════════

    # Scheduling requests from non-close relationships
    if any(word in email.body.lower() for word in ['meeting', 'schedule', 'calendar']):
        if features.relationship_strength < 0.5:
            return 'ai_partial', {'action': 'draft_schedule', 'needs_approval': True}

    # Confirmation requests from services/weak relationships
    if 'confirm' in email.body.lower() and features.relationship_strength < 0.3:
        return 'ai_partial', {'action': 'draft_confirmation', 'needs_approval': True}

    # ═══════════════════════════════════════════════════════════════
    # HUMAN_REQUIRED: Only human can handle
    # ═══════════════════════════════════════════════════════════════

    # Approval/decision requests - always need human
    if any(word in email.body.lower() for word in ['approve', 'approval', 'decision', 'sign off']):
        return 'human_required', {'reason': 'decision_needed', 'priority': 'high'}

    # High-relationship personal emails - human connection matters
    if features.relationship_strength > 0.6 and not features.is_service_email:
        return 'human_required', {'reason': 'important_relationship'}

    # Explicit reply requests from real people
    if not features.is_service_email:
        if any(phrase in email.body.lower() for phrase in ['let me know', 'please reply', 'get back to me']):
            return 'human_required', {'reason': 'explicit_reply_request'}

    # ═══════════════════════════════════════════════════════════════
    # NEEDS_LLM: Can't determine from rules alone
    # ═══════════════════════════════════════════════════════════════

    return 'needs_llm', {'reason': 'ambiguous'}
```

### Output: `email_ai_classification` Table

```sql
CREATE TABLE email_ai_classification (
    email_id INTEGER PRIMARY KEY REFERENCES emails(id),

    -- Rule-based classification (pre-LLM)
    predicted_handleability TEXT,      -- 'ai_full', 'ai_partial', 'human_required', 'needs_llm'
    classification_reason TEXT,        -- Why this classification
    classification_metadata JSONB,     -- Additional context (suggested folder, etc.)

    -- Sub-patterns detected
    has_question BOOLEAN,
    has_request_words BOOLEAN,
    has_scheduling_words BOOLEAN,
    has_deadline_words BOOLEAN,
    has_approval_words BOOLEAN,
    is_newsletter_pattern BOOLEAN,
    is_fyi_pattern BOOLEAN,

    -- LLM processing flags
    needs_llm_classification BOOLEAN,  -- Should Phase 1 LLM run?
    needs_llm_deep_extraction BOOLEAN, -- Should Phase 2 LLM run?
    llm_priority INTEGER,              -- Priority for LLM processing (1-5)

    created_at TIMESTAMP DEFAULT NOW()
);
```

### Cost: FREE (local pattern matching)

---

## Multi-Phase LLM Extraction Strategy

### The Problem with Single-Pass LLM

Original plan: Run full extraction on ~4,400 emails = ~$3-8

**Issues**:
1. Full extraction is overkill for many emails (newsletters just need "file it")
2. Some emails need deep context that requires expensive processing
3. Actual user behavior should inform what's worth extracting

### The Multi-Phase Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-PHASE LLM EXTRACTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ALL EMAILS (22,618)                                                        │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 0: Rule-Based Classification (FREE)                            │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ • ai_full (30.9%) → Skip LLM, just file                             │   │
│  │ • ai_partial (17.2%) → Light LLM for draft suggestions              │   │
│  │ • human_required (7.4%) → Full LLM for context extraction           │   │
│  │ • needs_llm (44.6%) → Run Phase 1 classification                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: Quick Classification (Haiku, ~$1-2)                         │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Target: ~10,000 'needs_llm' emails                                  │   │
│  │ Extract: action_type, urgency, ai_handleability (simple fields)     │   │
│  │ Purpose: Decide which emails need deep extraction                   │   │
│  │                                                                     │   │
│  │ Output categories:                                                  │   │
│  │ • 'skip' → Auto-file, no further processing                         │   │
│  │ • 'quick_action' → Simple AI action, no deep context needed        │   │
│  │ • 'needs_context' → Proceed to Phase 2                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2: Deep Extraction (Haiku/Sonnet, ~$2-4)                       │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Target: ~2,000-3,000 high-value emails                              │   │
│  │ • human_required (1,674)                                            │   │
│  │ • Phase 1 'needs_context' (~1,000-2,000)                            │   │
│  │ • User replied quickly (training signal)                            │   │
│  │ • High priority + ignored (learn why)                               │   │
│  │                                                                     │   │
│  │ Extract: Full 7-dimension schema                                    │   │
│  │ • Task details, deadlines, dependencies                             │   │
│  │ • Context requirements (person, project, thread)                    │   │
│  │ • AI capability assessment                                          │   │
│  │ • Briefing one-liner and key points                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 3: Cluster Propagation (FREE)                                  │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ For emails similar to Phase 2 emails:                               │   │
│  │ • Find nearest neighbors by embedding                               │   │
│  │ • Propagate labels from extracted emails                            │   │
│  │ • Validate against actual user behavior                             │   │
│  │                                                                     │   │
│  │ Extends coverage from 3,000 → 15,000+ emails                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Quick Classification Prompt

**Goal**: Simple classification to route emails, not full extraction.

```markdown
Classify this email for an AI assistant. Respond with JSON only.

Email:
From: {sender}
Subject: {subject}
Preview: {first_500_chars}

Respond with:
{
  "action_type": "reply|task|decision|fyi|none",
  "urgency": "immediate|today|this_week|whenever|none",
  "ai_can_handle": "fully|partially|not_at_all",
  "next_step": "skip|quick_action|needs_context",
  "quick_action": "file|acknowledge|schedule|forward|null",
  "one_liner": "Brief summary (10 words max)"
}
```

**Cost**: ~100 tokens per email × 10,000 emails = ~$0.25-0.50 (Haiku)

### Phase 2: Deep Extraction (Selective)

**Selection criteria**:
1. `human_required` from Phase 0 (all 1,674)
2. `needs_context` from Phase 1 (~1,500)
3. User replied within 1 hour (training: what made it urgent?)
4. High priority score but ignored (training: what made it ignorable?)

**Prompt**: Use full 7-dimension schema from Phase 4C design.

**Cost**: ~500 tokens per email × 3,000 emails = ~$1.50-3.00 (Haiku)

### Phase 3: Cluster Propagation

For emails NOT processed by LLM, use embedding similarity to propagate labels:

```python
def propagate_labels(unprocessed_emails, processed_emails):
    """
    For each unprocessed email, find similar processed emails
    and inherit their labels.
    """
    for email in unprocessed_emails:
        # Find 5 nearest neighbors by embedding
        neighbors = find_nearest_by_embedding(email, processed_emails, k=5)

        # If neighbors agree (>80% same label), propagate
        if label_agreement(neighbors) > 0.8:
            email.predicted_label = majority_label(neighbors)
            email.label_confidence = label_agreement(neighbors)
            email.label_source = 'propagated'
        else:
            # Neighbors disagree - this is an interesting case
            email.predicted_label = None
            email.needs_human_review = True
```

**Validation**: Compare propagated labels against actual user behavior.

### Expected Outcomes

| Phase | Emails | LLM Calls | Cost | Coverage |
|-------|--------|-----------|------|----------|
| Phase 0 | 22,618 | 0 | FREE | 55% classified |
| Phase 1 | ~10,000 | 10,000 | ~$0.50 | +45% classified |
| Phase 2 | ~3,000 | 3,000 | ~$2.00 | Full extraction |
| Phase 3 | ~15,000 | 0 | FREE | Labels propagated |
| **Total** | 22,618 | 13,000 | ~$2.50 | 100% |

### Key Insight: What AI COULD Do vs What User Did

**The goal is not to replicate user behavior.** The goal is to understand:

1. **For each email**: What COULD AI have done?
   - Fully handle? (file, auto-reply, schedule)
   - Partially help? (draft, prepare context, extract tasks)
   - Nothing? (human decision required)

2. **The user took all actions**: We're not training AI to predict user actions.
   We're training AI to understand which emails need human attention.

3. **Learning from behavior**: User behavior tells us what mattered:
   - Quick reply → This email deserved human attention
   - Ignored → AI could have handled this
   - Slow reply → Borderline case, learn the nuance

---

## AI Action Toolkit: Getting to 80-90% Automation

### The Full Picture

Classification is not enough. For AI to handle 80-90% of emails, it needs:
1. **Tools** to take actions (APIs, capabilities)
2. **Information** to decide (email content + attachments + context)
3. **Approval flows** (auto-execute vs human-in-loop)

### Available Data for Decision Making

| Source | What We Have | How It Helps |
|--------|--------------|--------------|
| **Email body** | Full text, 22,618 emails | Understand request/intent |
| **Attachments** | 7,648 files stored on disk | Process PDFs, calendars, docs |
| **ICS files** | 1,046 calendar invites | Auto-add to calendar |
| **PDFs** | 1,110 documents | Extract text, find action items |
| **Word/Excel** | 107 documents | Extract data, find requests |
| **Sender context** | Relationship, history, cluster | Know importance level |
| **Thread context** | in_reply_to, thread_id | Understand conversation flow |

### AI Action Toolkit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI ACTION TOOLKIT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 1: AUTO-EXECUTE (No approval needed)                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  📁 FILE_TO_FOLDER                                                          │
│     Input: email_id, folder_name                                            │
│     Action: Move email to specified folder                                  │
│     Use case: Newsletters → Promotions, Receipts → Orders                   │
│                                                                             │
│  📅 PARSE_CALENDAR_INVITE                                                   │
│     Input: ICS attachment                                                   │
│     Action: Extract event details, check conflicts                          │
│     Output: {event_name, datetime, attendees, conflicts}                    │
│                                                                             │
│  📄 EXTRACT_PDF_TEXT                                                        │
│     Input: PDF attachment                                                   │
│     Action: OCR/extract text, summarize                                     │
│     Output: {text, summary, detected_action_items}                          │
│                                                                             │
│  🏷️ ADD_LABEL                                                               │
│     Input: email_id, labels[]                                               │
│     Action: Apply labels for organization                                   │
│                                                                             │
│  📋 EXTRACT_TASK                                                            │
│     Input: email_id                                                         │
│     Action: Parse task from email, add to task list (not committed)         │
│     Output: {task_description, deadline, source_email}                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 2: PREPARE + APPROVE (Human confirms before execution)                │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ✉️ DRAFT_REPLY                                                              │
│     Input: email_id, reply_type, key_points[]                               │
│     Action: Generate reply draft                                            │
│     Output: {draft_text, confidence, needs_personalization}                 │
│     Approval: Human reviews, edits, sends                                   │
│                                                                             │
│  📅 SCHEDULE_MEETING                                                        │
│     Input: email_id, suggested_times[], duration, attendees[]               │
│     Action: Find availability, propose times                                │
│     Output: {proposed_slots[], conflicts[], draft_invite}                   │
│     Approval: Human picks slot, confirms send                               │
│                                                                             │
│  ➡️ FORWARD_TO                                                               │
│     Input: email_id, recipient, context_note                                │
│     Action: Prepare forward with context                                    │
│     Output: {forward_draft, suggested_note}                                 │
│     Approval: Human confirms recipient and note                             │
│                                                                             │
│  📝 CREATE_TASK                                                             │
│     Input: email_id, task_details                                           │
│     Action: Create task in task manager                                     │
│     Output: {task_draft, deadline, project}                                 │
│     Approval: Human confirms task creation                                  │
│                                                                             │
│  📆 ACCEPT_CALENDAR_INVITE                                                  │
│     Input: email_id, ICS attachment                                         │
│     Action: Parse invite, check calendar                                    │
│     Output: {event_details, conflicts, tentative_accept}                    │
│     Approval: Human confirms acceptance                                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 3: CONTEXT PREPARATION (Human takes action with AI-prepared info)    │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  📊 PREPARE_DECISION_CONTEXT                                                │
│     Input: email_id                                                         │
│     Action: Gather related emails, attachments, history                     │
│     Output: {                                                               │
│       summary: "Contract renewal request",                                  │
│       background: "Previous contract terms...",                             │
│       attachments_summary: "PDF contains...",                               │
│       related_emails: [{id, summary}...],                                   │
│       decision_required: "Approve/reject renewal",                          │
│       options: ["Approve as-is", "Counter-offer", "Decline"]                │
│     }                                                                       │
│                                                                             │
│  🔍 SEARCH_RELATED                                                          │
│     Input: email_id                                                         │
│     Action: Find related emails by embedding similarity                     │
│     Output: {related_emails[], common_thread, project_context}              │
│                                                                             │
│  📎 SUMMARIZE_ATTACHMENT                                                    │
│     Input: attachment_id                                                    │
│     Action: Extract and summarize document content                          │
│     Output: {summary, key_points, action_items, entities}                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Attachment Processing Pipeline

```python
def process_attachment(attachment):
    """
    Extract actionable information from attachments.
    """
    if attachment.content_type in ['application/ics', 'text/calendar']:
        # Calendar invite - parse and extract event
        return {
            'type': 'calendar_invite',
            'action': 'ACCEPT_CALENDAR_INVITE',
            'data': parse_ics(attachment.stored_path),
            # Returns: event_name, datetime, location, attendees
        }

    elif attachment.content_type == 'application/pdf':
        # PDF - extract text and find action items
        text = extract_pdf_text(attachment.stored_path)
        return {
            'type': 'document',
            'action': 'SUMMARIZE_ATTACHMENT',
            'data': {
                'text': text,
                'summary': summarize_text(text),
                'action_items': extract_action_items(text),
                'contains_signature_request': 'sign' in text.lower(),
                'contains_deadline': detect_deadline(text),
            }
        }

    elif attachment.content_type in ['application/msword', '...docx']:
        # Word doc - similar to PDF
        text = extract_docx_text(attachment.stored_path)
        return {...}

    elif attachment.content_type in ['image/png', 'image/jpeg']:
        # Image - usually not actionable (logos, signatures)
        # Could OCR if needed
        return {
            'type': 'image',
            'action': None,  # Usually no action needed
            'data': {'needs_ocr': is_likely_document(attachment)}
        }
```

### Email → Action Mapping

Based on our data, here's how emails map to actions:

| Email Pattern | Count | % | AI Action | Approval |
|---------------|-------|---|-----------|----------|
| Newsletter/Marketing | 6,983 | 30.9% | FILE_TO_FOLDER | Auto |
| Calendar invite (ICS) | 1,046 | 4.6% | ACCEPT_CALENDAR_INVITE | Approve |
| Order confirmation | ~1,000 | 4.4% | FILE_TO_FOLDER + EXTRACT_TASK | Auto |
| Meeting request (text) | ~500 | 2.2% | SCHEDULE_MEETING | Approve |
| FYI from contact | ~2,000 | 8.8% | DRAFT_REPLY (acknowledgment) | Approve |
| Question needing answer | ~1,500 | 6.6% | PREPARE_DECISION_CONTEXT | Human |
| Approval request | ~350 | 1.5% | PREPARE_DECISION_CONTEXT | Human |
| Task assignment | ~500 | 2.2% | CREATE_TASK + DRAFT_REPLY | Approve |
| PDF attachment | 1,110 | 4.9% | SUMMARIZE_ATTACHMENT | Auto |
| General person email | ~8,000 | 35% | Varies by content | Varies |

### Getting to 80-90%

**Tier 1 (Auto-execute)**: ~40% of emails
- Newsletters, notifications → Auto-file
- Calendar accepts → Auto-file
- Receipts → Auto-file + extract

**Tier 2 (Prepare + Approve)**: ~40% of emails
- Meeting requests → Draft response, await approval
- FYI from contacts → Draft acknowledgment
- Task assignments → Extract task, draft confirm

**Tier 3 (Context prep)**: ~20% of emails
- Decisions/approvals → Prepare context, human decides
- Complex requests → Summarize, human handles

**Total AI involvement**: 80-100% (human only writes 10-20% from scratch)

### LLM Extraction: What to Capture

For each email, LLM should extract:

```json
{
  "email_id": 1234,

  "understanding": {
    "intent": "Meeting request for project kickoff",
    "sender_wants": "Schedule 1-hour meeting next week",
    "attachments_contain": "PDF with project brief"
  },

  "suggested_action": {
    "primary_tool": "SCHEDULE_MEETING",
    "tier": 2,
    "confidence": 0.85,
    "parameters": {
      "duration": 60,
      "suggested_times": ["next Tuesday 2pm", "next Wednesday 10am"],
      "attendees": ["sender"],
      "topic": "Project kickoff"
    }
  },

  "attachment_actions": [
    {
      "attachment_id": 567,
      "tool": "SUMMARIZE_ATTACHMENT",
      "summary": "Project brief: 3-month timeline, $50k budget, key deliverables..."
    }
  ],

  "draft_outputs": {
    "reply_draft": "Hi [Name], I'd be happy to meet. How about Tuesday at 2pm or Wednesday at 10am? I've reviewed the project brief...",
    "task_draft": {
      "title": "Project kickoff meeting",
      "due": "next week",
      "context": "Review project brief before meeting"
    }
  },

  "human_required": {
    "needed": true,
    "reason": "Confirm availability and meeting time",
    "decision_type": "schedule_selection"
  }
}
```

---

## Phase 4C: LLM Feature Extraction (AI Assistant Pipeline)

### What It Does
For selected emails, use LLM to extract everything needed for an AI assistant to:
1. **Understand** what the email requires
2. **Act** on what it can handle autonomously
3. **Brief** the user on what needs their attention

### Design Philosophy: The AI Assistant Workflow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        AI EMAIL ASSISTANT WORKFLOW                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Email arrives                                                             │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │ 1. ANALYZE       │  What is this? Who sent it? What do they want?       │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │ 2. CLASSIFY      │  Action needed? Task? FYI? Schedulable?              │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │ 3. ASSESS AI     │  What can AI do? Draft? Schedule? Forward?           │
│  │    CAPABILITY    │  What MUST be human? Decision? Approval?             │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │ 4. EXTRACT       │  Context user needs: project background,             │
│  │    CONTEXT       │  person relationship, relevant history               │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    5. ROUTE TO OUTCOME                            │      │
│  ├──────────────────────────────────────────────────────────────────┤      │
│  │                                                                   │      │
│  │  AI_HANDLES          │  NEEDS_USER          │  FYI_ONLY          │      │
│  │  ────────────        │  ──────────          │  ────────          │      │
│  │  • Schedule meeting  │  • Decision needed   │  • Newsletter      │      │
│  │  • Send follow-up    │  • Personal reply    │  • Notification    │      │
│  │  • File/organize     │  • Approval required │  • Status update   │      │
│  │  • Draft response    │  • Sensitive topic   │  • Confirmation    │      │
│  │                      │                      │                    │      │
│  │  → AI executes       │  → Brief user with   │  → Log for user    │      │
│  │    (with approval    │    full context      │    if they want    │      │
│  │    if configured)    │                      │    to review       │      │
│  │                                                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### The Seven Extraction Dimensions

#### 1. Action Detection
**Question**: Is the user being asked to take an action?

```yaml
action_required:
  is_action_requested: boolean      # Someone explicitly asking for something?
  action_type: enum                 # reply | task | decision | approval | fyi | none
  action_explicit: boolean          # "Please send X" vs implied expectation
  action_description: string        # "Schedule meeting with vendor"
```

#### 2. Task Extraction
**Question**: Is there a task? What needs to happen?

```yaml
task:
  has_task: boolean                 # Is there a discrete task here?
  task_description: string          # "Review contract and sign"
  deadline: string                  # "by Friday", "ASAP", "when convenient"
  deadline_type: enum               # hard | soft | none
  deadline_parsed: timestamp        # Actual date if parseable
  dependencies: string[]            # "After John sends the doc"
  deliverable: string               # What's the output? "Signed PDF"
```

#### 3. Reply Requirements
**Question**: Does this need a reply? What kind?

```yaml
reply:
  needs_reply: boolean              # Is a response expected?
  reply_urgency: enum               # immediate | same_day | this_week | whenever | none
  reply_type: enum                  # answer_question | confirm | provide_info | decision | none
  reply_content_needed: string[]    # What info needed to reply? ["project status", "availability"]
  can_be_brief: boolean             # "Yes" vs detailed response needed
```

#### 4. Context Requirements
**Question**: What does user need to know to take action?

```yaml
context:
  # Person context
  sender_context_needed: boolean    # Need to know about this person?
  sender_context: string            # "VP at client company, working on Project X"
  relationship_relevant: boolean    # Does relationship matter for response?

  # Project context
  project_mentioned: string         # Project/initiative referenced
  project_context_needed: boolean   # Need project background?
  project_context: string           # "Q1 launch, currently in testing phase"

  # Historical context
  thread_context: string            # What happened before in this thread?
  related_emails_hint: string       # "See emails from last week about budget"

  # Decision context
  decision_background: string       # If decision needed, what's the context?
  options_presented: string[]       # Choices being offered
```

#### 5. AI Capability Assessment
**Question**: What can AI do? What must be human?

```yaml
ai_capability:
  # What AI can handle
  ai_can_handle: enum               # fully | partially | not_at_all
  ai_actions_possible: string[]     # ["draft_reply", "schedule_meeting", "forward_to_X"]
  ai_draft_appropriate: boolean     # Should AI draft a response?
  ai_draft_template: string         # If yes, what kind? "Confirmation", "Decline", etc.

  # What needs human
  requires_human_because: string[]  # ["decision", "personal_relationship", "sensitive"]
  human_decision_type: string       # "Approve budget" / "Choose vendor" / etc.
  human_input_needed: string[]      # Specific things human must provide

  # Suggested AI actions
  suggested_ai_actions:
    - action: string                # "schedule_meeting"
      confidence: float             # 0.9
      requires_approval: boolean    # true
      details: string               # "Meeting with John, next Tuesday 2pm suggested"
```

#### 6. Priority & Scheduling
**Question**: How urgent? Can it be scheduled/batched?

```yaml
priority:
  # Urgency assessment
  urgency_level: enum               # critical | high | normal | low | none
  urgency_reason: string            # "Deadline tomorrow" / "From CEO" / etc.
  time_sensitive: boolean           # Will waiting cause problems?
  time_sensitivity_window: string   # "Must respond by EOD" / "Within 48 hours"

  # Scheduling potential
  can_be_scheduled: boolean         # Can this wait for a scheduled time?
  suggested_handling_time: enum     # now | today | this_week | batch_weekly
  can_batch_with: string            # "Other vendor emails" / "Weekly review"

  # Attention allocation
  attention_level_needed: enum      # focused | quick_scan | background
  estimated_handling_time: string   # "2 min reply" / "30 min review"
```

#### 7. Briefing Classification
**Question**: How should AI present this to user?

```yaml
briefing:
  # Category for user digest
  category: enum                    # action_required | awaiting_input | fyi | ai_handled

  # Briefing priority (for ordering in digest)
  briefing_priority: int            # 1-5, 1 = show first

  # One-line summary for digest
  one_liner: string                 # "John needs budget approval for Q1 campaign by Friday"

  # What user needs to know
  key_points: string[]              # Bullet points for quick scan

  # What user needs to do
  user_action_summary: string       # "Reply with approved/denied + any conditions"

  # What AI did/can do
  ai_action_summary: string         # "Drafted approval response, ready to send"
```

### Complete Extraction Schema

```sql
CREATE TABLE llm_analysis (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id) UNIQUE,

    -- ═══════════════════════════════════════════════════════════════════
    -- 1. ACTION DETECTION
    -- ═══════════════════════════════════════════════════════════════════
    action_requested BOOLEAN,
    action_type TEXT,                   -- 'reply', 'task', 'decision', 'approval', 'fyi', 'none'
    action_explicit BOOLEAN,
    action_description TEXT,

    -- ═══════════════════════════════════════════════════════════════════
    -- 2. TASK EXTRACTION
    -- ═══════════════════════════════════════════════════════════════════
    has_task BOOLEAN,
    task_description TEXT,
    task_deadline TEXT,                 -- Raw deadline text
    task_deadline_type TEXT,            -- 'hard', 'soft', 'none'
    task_deadline_parsed TIMESTAMP,     -- Parsed date if possible
    task_dependencies TEXT[],
    task_deliverable TEXT,

    -- ═══════════════════════════════════════════════════════════════════
    -- 3. REPLY REQUIREMENTS
    -- ═══════════════════════════════════════════════════════════════════
    needs_reply BOOLEAN,
    reply_urgency TEXT,                 -- 'immediate', 'same_day', 'this_week', 'whenever', 'none'
    reply_type TEXT,                    -- 'answer_question', 'confirm', 'provide_info', 'decision', 'none'
    reply_content_needed TEXT[],        -- What info needed to reply
    reply_can_be_brief BOOLEAN,

    -- ═══════════════════════════════════════════════════════════════════
    -- 4. CONTEXT REQUIREMENTS
    -- ═══════════════════════════════════════════════════════════════════
    -- Person context
    sender_context_needed BOOLEAN,
    sender_context TEXT,
    relationship_relevant BOOLEAN,

    -- Project context
    project_mentioned TEXT,
    project_context_needed BOOLEAN,
    project_context TEXT,

    -- Historical context
    thread_context TEXT,
    related_emails_hint TEXT,

    -- Decision context
    decision_background TEXT,
    options_presented TEXT[],

    -- ═══════════════════════════════════════════════════════════════════
    -- 5. AI CAPABILITY
    -- ═══════════════════════════════════════════════════════════════════
    ai_can_handle TEXT,                 -- 'fully', 'partially', 'not_at_all'
    ai_actions_possible TEXT[],
    ai_draft_appropriate BOOLEAN,
    ai_draft_template TEXT,

    requires_human_because TEXT[],
    human_decision_type TEXT,
    human_input_needed TEXT[],

    -- Suggested actions (JSONB for flexibility)
    suggested_ai_actions JSONB,

    -- ═══════════════════════════════════════════════════════════════════
    -- 6. PRIORITY & SCHEDULING
    -- ═══════════════════════════════════════════════════════════════════
    urgency_level TEXT,                 -- 'critical', 'high', 'normal', 'low', 'none'
    urgency_reason TEXT,
    time_sensitive BOOLEAN,
    time_sensitivity_window TEXT,

    can_be_scheduled BOOLEAN,
    suggested_handling_time TEXT,       -- 'now', 'today', 'this_week', 'batch_weekly'
    can_batch_with TEXT,

    attention_level_needed TEXT,        -- 'focused', 'quick_scan', 'background'
    estimated_handling_time TEXT,

    -- ═══════════════════════════════════════════════════════════════════
    -- 7. BRIEFING CLASSIFICATION
    -- ═══════════════════════════════════════════════════════════════════
    briefing_category TEXT,             -- 'action_required', 'awaiting_input', 'fyi', 'ai_handled'
    briefing_priority INTEGER,          -- 1-5
    one_liner TEXT,
    key_points TEXT[],
    user_action_summary TEXT,
    ai_action_summary TEXT,

    -- ═══════════════════════════════════════════════════════════════════
    -- ENTITIES & TRAINING
    -- ═══════════════════════════════════════════════════════════════════
    mentioned_people TEXT[],
    mentioned_projects TEXT[],
    mentioned_deadlines TEXT[],
    mentioned_amounts TEXT[],

    -- Training signals (for RL)
    importance_signals TEXT[],
    ignore_signals TEXT[],

    -- ═══════════════════════════════════════════════════════════════════
    -- METADATA
    -- ═══════════════════════════════════════════════════════════════════
    model TEXT,
    prompt_version TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd FLOAT,
    processing_time_ms INTEGER,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_llm_briefing_category ON llm_analysis(briefing_category);
CREATE INDEX idx_llm_urgency ON llm_analysis(urgency_level);
CREATE INDEX idx_llm_ai_can_handle ON llm_analysis(ai_can_handle);
CREATE INDEX idx_llm_needs_reply ON llm_analysis(needs_reply);
```

### The Extraction Prompt

```markdown
You are analyzing an email to help an AI assistant manage a user's inbox.
Extract structured information to enable the assistant to:
1. Handle what it can autonomously
2. Brief the user on what needs their attention
3. Provide all context needed for the user to act quickly

## Email
From: {sender}
To: {recipient}
Subject: {subject}
Date: {date}
Thread context: {thread_summary}  # If available

---
{body}
---

## Known Context
- Sender relationship: {relationship_strength}
- Past interactions: {interaction_history}
- Projects mentioned: {known_projects}

## Extract the following (respond in JSON):

```json
{
  "action": {
    "is_action_requested": true/false,
    "action_type": "reply|task|decision|approval|fyi|none",
    "action_explicit": true/false,
    "action_description": "string or null"
  },

  "task": {
    "has_task": true/false,
    "task_description": "string or null",
    "deadline": "string or null",
    "deadline_type": "hard|soft|none",
    "dependencies": ["string"],
    "deliverable": "string or null"
  },

  "reply": {
    "needs_reply": true/false,
    "urgency": "immediate|same_day|this_week|whenever|none",
    "type": "answer_question|confirm|provide_info|decision|none",
    "content_needed": ["what info is needed to reply"],
    "can_be_brief": true/false
  },

  "context": {
    "sender_context_needed": true/false,
    "sender_context": "string or null",
    "relationship_relevant": true/false,
    "project_mentioned": "string or null",
    "project_context_needed": true/false,
    "project_context": "string or null",
    "thread_context": "string or null",
    "decision_background": "string or null",
    "options_presented": ["string"]
  },

  "ai_capability": {
    "can_handle": "fully|partially|not_at_all",
    "actions_possible": ["draft_reply", "schedule_meeting", "forward", "file", "none"],
    "draft_appropriate": true/false,
    "draft_template": "confirmation|decline|request_info|acknowledgment|none",
    "requires_human_because": ["decision", "personal", "sensitive", "expertise", "approval"],
    "human_decision_type": "string or null",
    "human_input_needed": ["string"],
    "suggested_actions": [
      {
        "action": "string",
        "confidence": 0.0-1.0,
        "requires_approval": true/false,
        "details": "string"
      }
    ]
  },

  "priority": {
    "urgency_level": "critical|high|normal|low|none",
    "urgency_reason": "string or null",
    "time_sensitive": true/false,
    "time_sensitivity_window": "string or null",
    "can_be_scheduled": true/false,
    "suggested_handling_time": "now|today|this_week|batch_weekly",
    "can_batch_with": "string or null",
    "attention_level_needed": "focused|quick_scan|background",
    "estimated_handling_time": "string"
  },

  "briefing": {
    "category": "action_required|awaiting_input|fyi|ai_handled",
    "priority": 1-5,
    "one_liner": "Brief summary for inbox digest",
    "key_points": ["bullet points for quick scan"],
    "user_action_summary": "What user needs to do (or null)",
    "ai_action_summary": "What AI did/can do (or null)"
  },

  "entities": {
    "people": ["names mentioned"],
    "projects": ["projects/initiatives"],
    "deadlines": ["deadline strings"],
    "amounts": ["money/quantities"]
  },

  "training_signals": {
    "importance_indicators": ["signals suggesting importance"],
    "ignore_indicators": ["signals suggesting low importance"]
  }
}
```

### Response Guidelines:
1. Be conservative with AI capability - when in doubt, say "partially" or "not_at_all"
2. For briefing.one_liner, write as if telling user "John needs X by Friday" not "The email discusses..."
3. List specific actions AI can take, not vague capabilities
4. If deadline can be parsed to a date, include it; otherwise leave as text
5. For requires_human_because, be specific: "decision about vendor selection" not just "decision"
```

### Example Extractions

#### Example 1: Meeting Request (AI Can Handle)
```json
{
  "action": {
    "is_action_requested": true,
    "action_type": "task",
    "action_explicit": true,
    "action_description": "Schedule meeting to discuss Q1 results"
  },
  "task": {
    "has_task": true,
    "task_description": "Schedule 30-min meeting with Sarah",
    "deadline": "this week",
    "deadline_type": "soft",
    "deliverable": "Calendar invite sent"
  },
  "reply": {
    "needs_reply": true,
    "urgency": "this_week",
    "type": "confirm",
    "content_needed": ["available times"],
    "can_be_brief": true
  },
  "ai_capability": {
    "can_handle": "fully",
    "actions_possible": ["schedule_meeting", "draft_reply"],
    "draft_appropriate": true,
    "draft_template": "confirmation",
    "suggested_actions": [
      {
        "action": "schedule_meeting",
        "confidence": 0.95,
        "requires_approval": true,
        "details": "30-min meeting with Sarah Chen, suggested: Tue 2pm or Wed 10am"
      }
    ]
  },
  "priority": {
    "urgency_level": "normal",
    "can_be_scheduled": true,
    "suggested_handling_time": "today",
    "attention_level_needed": "quick_scan",
    "estimated_handling_time": "2 min"
  },
  "briefing": {
    "category": "ai_handled",
    "priority": 3,
    "one_liner": "Sarah wants to meet about Q1 results - I can schedule this",
    "key_points": ["Meeting request from Sarah Chen", "Topic: Q1 results review", "Duration: 30 min"],
    "ai_action_summary": "Ready to send invite for Tue 2pm, awaiting your approval"
  }
}
```

#### Example 2: Decision Required (Human Must Act)
```json
{
  "action": {
    "is_action_requested": true,
    "action_type": "decision",
    "action_explicit": true,
    "action_description": "Approve or reject vendor contract renewal"
  },
  "task": {
    "has_task": true,
    "task_description": "Review and approve vendor contract",
    "deadline": "Friday 5pm",
    "deadline_type": "hard",
    "deliverable": "Signed approval or rejection with feedback"
  },
  "reply": {
    "needs_reply": true,
    "urgency": "same_day",
    "type": "decision",
    "content_needed": ["approval decision", "any conditions or changes"],
    "can_be_brief": false
  },
  "context": {
    "sender_context_needed": true,
    "sender_context": "Procurement team lead handling vendor relationships",
    "project_mentioned": "Annual vendor renewals",
    "project_context": "Part of cost reduction initiative, 3 vendors up for renewal",
    "decision_background": "Contract increased 15% from last year",
    "options_presented": ["Approve as-is", "Approve with conditions", "Reject and re-negotiate"]
  },
  "ai_capability": {
    "can_handle": "partially",
    "actions_possible": ["draft_reply"],
    "draft_appropriate": false,
    "requires_human_because": ["budget decision", "contract commitment", "requires domain expertise"],
    "human_decision_type": "Vendor contract approval with budget impact",
    "human_input_needed": ["approve/reject decision", "any conditions to include"],
    "suggested_actions": [
      {
        "action": "prepare_context",
        "confidence": 0.9,
        "requires_approval": false,
        "details": "Pulled last year's contract for comparison, vendor performance summary attached"
      }
    ]
  },
  "priority": {
    "urgency_level": "high",
    "urgency_reason": "Hard deadline Friday, contract expires Monday",
    "time_sensitive": true,
    "time_sensitivity_window": "Must respond by Friday 5pm",
    "can_be_scheduled": false,
    "suggested_handling_time": "today",
    "attention_level_needed": "focused",
    "estimated_handling_time": "15 min review + decision"
  },
  "briefing": {
    "category": "action_required",
    "priority": 1,
    "one_liner": "Vendor contract renewal needs your approval by Friday - 15% increase from last year",
    "key_points": [
      "Acme Corp contract up for renewal",
      "15% cost increase vs last year",
      "Deadline: Friday 5pm (hard)",
      "Options: approve, approve w/ conditions, reject"
    ],
    "user_action_summary": "Review contract, decide: approve/conditions/reject",
    "ai_action_summary": "Prepared comparison with last year's contract and vendor performance summary"
  }
}
```

#### Example 3: FYI / Newsletter (Minimal Attention)
```json
{
  "action": {
    "is_action_requested": false,
    "action_type": "fyi",
    "action_explicit": false
  },
  "task": {
    "has_task": false
  },
  "reply": {
    "needs_reply": false,
    "urgency": "none"
  },
  "ai_capability": {
    "can_handle": "fully",
    "actions_possible": ["file", "summarize"],
    "suggested_actions": [
      {
        "action": "file_to_folder",
        "confidence": 0.95,
        "requires_approval": false,
        "details": "Filed to 'Industry News' folder"
      }
    ]
  },
  "priority": {
    "urgency_level": "none",
    "can_be_scheduled": true,
    "suggested_handling_time": "batch_weekly",
    "can_batch_with": "Other newsletters",
    "attention_level_needed": "background",
    "estimated_handling_time": "skip or 30 sec scan"
  },
  "briefing": {
    "category": "fyi",
    "priority": 5,
    "one_liner": "Industry newsletter - no action needed",
    "key_points": ["Weekly tech digest", "3 articles about AI trends"],
    "ai_action_summary": "Filed to Industry News, summarized headlines"
  }
}
```

### Cost Optimization

| Strategy | Savings |
|----------|---------|
| Only process selected emails (20%) | 80% |
| Use Haiku for most, Sonnet for complex decisions | 50% |
| Batch similar emails (same sender/cluster) | 30% |
| Skip newsletters/marketing (rule-based) | 20% |

**Estimated cost for 22K emails**: ~$3-8 (processing ~4,400 selected emails)

### User Briefing Interface

The end goal: AI processes emails and presents a structured briefing to the user.

#### Daily Digest Format

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         YOUR EMAIL BRIEFING                                │
│                         Monday, Jan 6, 2026                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ⚡ NEEDS YOUR ATTENTION (3 items)                                         │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  1. [HIGH] Vendor contract renewal - approval needed by Friday             │
│     From: Sarah Chen (Procurement)                                         │
│     Context: 15% increase from last year, part of cost reduction init      │
│     You need to: Decide approve/reject/conditions                          │
│     AI prepared: Comparison with last year's contract                      │
│     [View Details] [Draft Approval] [Draft Rejection]                      │
│                                                                            │
│  2. [HIGH] Q1 budget review meeting - decision on marketing spend          │
│     From: John Lee (Finance)                                               │
│     Context: Following up on last week's projection discussion             │
│     You need to: Confirm final numbers before board meeting                │
│     [View Details] [Review Numbers]                                        │
│                                                                            │
│  3. [NORMAL] Client feedback on prototype - response expected              │
│     From: Mike Johnson (Acme Corp - key account)                           │
│     Context: Phase 2 delivery, they have concerns about timeline           │
│     You need to: Address timeline concerns, confirm next steps             │
│     AI can: Draft acknowledgment, propose revised timeline                 │
│     [View Details] [Use AI Draft] [Write Custom]                           │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  🤖 AI HANDLED (5 items) - Review if you want                              │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  ✓ Meeting scheduled: Sarah Chen re: Q1 results (Tue 2pm)                  │
│  ✓ Meeting scheduled: Team standup moved to Wed 9am per request            │
│  ✓ Confirmation sent: Acknowledged receipt of contract docs                │
│  ✓ Follow-up sent: Reminded vendor about pending invoice                   │
│  ✓ Filed: 3 newsletters to "Industry News" folder                          │
│                                                                            │
│  [View All] [Undo Any]                                                     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  📋 TASKS EXTRACTED (2 new)                                                │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  □ Review vendor contract (Due: Fri) - from Sarah's email                  │
│  □ Prepare board presentation (Due: Next Tue) - from John's email          │
│                                                                            │
│  [Add to Task Manager] [Dismiss]                                           │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  📰 FYI - SKIM WHEN YOU HAVE TIME (12 items)                               │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  • Industry newsletter: 3 AI trends articles                               │
│  • Company announcement: New parking policy                                │
│  • Project update: Backend migration 80% complete                          │
│  • ... and 9 more                                                          │
│                                                                            │
│  [View All FYI]                                                            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  📊 SUMMARY                                                                │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  • 47 emails received today                                                │
│  • 3 need your attention (est. 25 min total)                               │
│  • 5 handled by AI                                                         │
│  • 12 FYI (can batch-review weekly)                                        │
│  • 27 low-priority (auto-filed)                                            │
│                                                                            │
│  Time saved: ~45 min                                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### Briefing Data Structure

```sql
-- Aggregated briefing view (generated from llm_analysis)
CREATE VIEW user_briefing AS
SELECT
    la.briefing_category,
    la.briefing_priority,
    la.one_liner,
    la.key_points,
    la.user_action_summary,
    la.ai_action_summary,
    la.urgency_level,
    la.estimated_handling_time,

    -- Email context
    e.id as email_id,
    e.sender,
    e.subject,
    e.received_at,

    -- Relationship context
    ef.relationship_strength,
    ef.sender_name,

    -- AI actions available
    la.ai_can_handle,
    la.suggested_ai_actions

FROM llm_analysis la
JOIN emails e ON e.id = la.email_id
LEFT JOIN email_features ef ON ef.email_id = e.id
ORDER BY
    CASE la.briefing_category
        WHEN 'action_required' THEN 1
        WHEN 'awaiting_input' THEN 2
        WHEN 'ai_handled' THEN 3
        WHEN 'fyi' THEN 4
    END,
    la.briefing_priority,
    la.urgency_level DESC;
```

#### API Response Format

```json
{
  "briefing_date": "2026-01-06",
  "summary": {
    "total_emails": 47,
    "needs_attention": 3,
    "ai_handled": 5,
    "fyi": 12,
    "auto_filed": 27,
    "estimated_attention_time_min": 25,
    "time_saved_min": 45
  },
  "sections": {
    "needs_attention": [
      {
        "email_id": 1234,
        "priority": 1,
        "urgency": "high",
        "one_liner": "Vendor contract renewal - approval needed by Friday",
        "sender": {"name": "Sarah Chen", "role": "Procurement", "relationship": "colleague"},
        "context": "15% increase from last year, part of cost reduction initiative",
        "user_action": "Decide approve/reject/conditions",
        "ai_prepared": "Comparison with last year's contract",
        "estimated_time": "15 min",
        "actions": ["view_details", "draft_approval", "draft_rejection"]
      }
    ],
    "ai_handled": [
      {
        "email_id": 2345,
        "action_taken": "schedule_meeting",
        "summary": "Meeting scheduled: Sarah Chen re: Q1 results (Tue 2pm)",
        "reversible": true
      }
    ],
    "tasks_extracted": [
      {
        "email_id": 1234,
        "task": "Review vendor contract",
        "deadline": "2026-01-10",
        "source_sender": "Sarah Chen"
      }
    ],
    "fyi": [
      {
        "email_id": 3456,
        "one_liner": "Industry newsletter: 3 AI trends articles",
        "can_batch_with": "Other newsletters"
      }
    ]
  }
}
```

#### Interaction Modes

| Mode | User Experience | When to Use |
|------|-----------------|-------------|
| **Morning Briefing** | Full digest with all sections | Start of day |
| **Quick Check** | Only "Needs Attention" count + top item | During meetings |
| **Deep Review** | Expand all FYI, show AI reasoning | End of day |
| **Approval Mode** | Step through AI-pending actions | Batch approve/reject |

#### AI Action Approval Flow

```
AI wants to: Schedule meeting with Sarah Chen for Tue 2pm

[Approve] - AI sends invite immediately
[Modify] - "Make it Wed instead" → AI reschedules
[Reject] - AI marks as "needs human handling"
[Approve All Similar] - Auto-approve future meeting scheduling
```

#### Learning from Briefing Interactions

Every user action on the briefing becomes training data:

| User Action | Training Signal |
|-------------|-----------------|
| Clicked "View Details" | Email was important enough to read |
| Used AI draft | AI assessment was accurate |
| Wrote custom reply | AI draft insufficient (learn why) |
| Dismissed task | Task extraction was wrong |
| Undid AI action | AI overstepped |
| Approved AI action | AI correctly autonomous |
| Marked FYI as important | Misclassified, should be higher |

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
