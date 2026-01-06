# Phase 2: Basic ML Features - Proposal

## What We're Building

A feature extraction system that computes **relationship-based and behavioral metrics** for each email, enabling smart ranking and filtering without needing LLMs or embeddings.

## The Problem We're Solving

Right now we have action labels (REPLIED, ARCHIVED, etc.) but we can't answer:
- Which senders are most important to me?
- Which emails need immediate attention vs can wait?
- Is this sender someone I engage with regularly?
- Is this a service email or a real person?

## What Phase 2 Adds

### Creates: `email_features` Table

A new table with **30 computed features** per email that can be queried instantly:

```sql
CREATE TABLE email_features (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id),

    -- Relationship Features (11 dimensions)
    emails_from_sender_7d INTEGER,      -- Recent activity
    emails_from_sender_30d INTEGER,
    emails_from_sender_90d INTEGER,
    emails_from_sender_all INTEGER,
    user_replied_to_sender_count INTEGER,
    user_replied_to_sender_rate FLOAT,  -- 0.0 to 1.0
    avg_response_time_hours FLOAT,
    user_initiated_ratio FLOAT,         -- Who starts conversations
    days_since_last_interaction INTEGER,
    sender_replies_to_you_rate FLOAT,   -- Do they reply when you email them?
    relationship_strength FLOAT,        -- Overall score 0-1

    -- Service Detection (7 dimensions)
    is_service_email BOOLEAN,
    service_confidence FLOAT,           -- How sure we are
    service_type TEXT,                  -- newsletter, transactional, etc.
    service_importance FLOAT,           -- 0-1: Is this service email important? (NEW)
    has_unsubscribe_link BOOLEAN,
    has_list_unsubscribe_header BOOLEAN,
    from_common_service_domain BOOLEAN, -- @noreply.*, @no-reply.*, etc.

    -- Temporal Features (8 dimensions)
    hour_of_day INTEGER,                -- 0-23
    day_of_week INTEGER,                -- 0-6 (Monday=0)
    is_weekend BOOLEAN,
    is_business_hours BOOLEAN,          -- 9am-5pm local time
    days_since_received INTEGER,
    is_recent BOOLEAN,                  -- < 7 days old
    time_bucket TEXT,                   -- morning/afternoon/evening/night
    urgency_score FLOAT,                -- 0-1 based on timing + relationship

    -- Content Basic (5 dimensions)
    subject_word_count INTEGER,
    body_word_count INTEGER,            -- HTML stripped, plain text only
    has_attachments BOOLEAN,
    attachment_count INTEGER,
    recipient_count INTEGER,            -- to + cc

    created_at TIMESTAMP DEFAULT NOW()
);
```

### Note: HTML Content Handling

**Problem:** Email bodies often contain HTML which inflates word counts and makes text extraction unreliable.

**Solution:** We strip HTML before computing `body_word_count`:
```python
from bs4 import BeautifulSoup
import re

def extract_plain_text(html_body: str) -> str:
    """Extract plain text from HTML email body."""
    if not html_body:
        return ""

    # Parse HTML
    soup = BeautifulSoup(html_body, 'html.parser')

    # Remove script and style elements
    for element in soup(['script', 'style', 'head', 'meta']):
        element.decompose()

    # Get text
    text = soup.get_text(separator=' ')

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
```

This ensures:
- Marketing emails with heavy HTML don't get inflated word counts
- Plain text emails are counted accurately
- We compare apples to apples across email types

## How This Helps You

### 1. **Smart Email Ranking**

Query to find "emails I should read first":

```sql
SELECT e.subject, e.from_email,
       ef.relationship_strength,
       ef.urgency_score,
       ef.service_type
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE e.action = 'PENDING'
  AND ef.is_service_email = FALSE
ORDER BY (ef.relationship_strength * ef.urgency_score) DESC
LIMIT 20;
```

This surfaces:
- Real people (not newsletters)
- People you have strong relationships with
- Recent/urgent emails

### 2. **Identify Email Patterns**

See who drains your time:

```sql
-- High-volume senders you rarely reply to
SELECT from_email,
       emails_from_sender_all,
       user_replied_to_sender_rate,
       service_type
FROM email_features ef
JOIN emails e ON e.id = ef.email_id
WHERE emails_from_sender_all > 50
  AND user_replied_to_sender_rate < 0.1
GROUP BY from_email, emails_from_sender_all, user_replied_to_sender_rate, service_type
ORDER BY emails_from_sender_all DESC;
```

### 3. **Find Neglected Relationships**

People you used to engage with but haven't recently:

```sql
SELECT e.from_email,
       ef.emails_from_sender_all,
       ef.days_since_last_interaction,
       ef.user_replied_to_sender_rate
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.relationship_strength > 0.7  -- Was strong
  AND ef.days_since_last_interaction > 30
  AND ef.user_replied_to_sender_rate > 0.3
GROUP BY e.from_email, ef.emails_from_sender_all,
         ef.days_since_last_interaction, ef.user_replied_to_sender_rate
ORDER BY ef.relationship_strength DESC;
```

### 4. **Service Email Cleanup**

Bulk archive newsletters you never read:

```sql
-- Find service emails you've never engaged with
SELECT e.from_email,
       COUNT(*) as email_count,
       ef.service_type,
       MAX(ef.has_unsubscribe_link) as has_unsub
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.is_service_email = TRUE
  AND e.action IN ('ARCHIVED', 'IGNORED')
GROUP BY e.from_email, ef.service_type
HAVING COUNT(*) > 10
ORDER BY email_count DESC;
```

### 5. **Response Time Analysis**

When do you respond fastest?

```sql
SELECT ef.time_bucket,
       AVG(e.response_time_seconds)/3600.0 as avg_hours
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE e.action = 'REPLIED'
  AND ef.time_bucket IS NOT NULL
GROUP BY ef.time_bucket
ORDER BY avg_hours;
```

## Feature Computation Logic

### Relationship Strength (0.0 to 1.0)

Combines multiple signals:
```python
relationship_strength = (
    0.35 * min(emails_from_sender_30d / 10, 1.0) +     # Frequency
    0.45 * user_replied_to_sender_rate +                # Engagement
    0.1 * sender_replies_to_you_rate +                 # Reciprocity
    0.05 * (1 - min(days_since_last_interaction/90, 1)) + # Recency
    0.05 * user_initiated_ratio                         # Balance
)
```

### Service Detection

Marks as service email if:
- Has "List-Unsubscribe" header
- Has "unsubscribe" link in body
- From common service domain (@noreply, @notifications, etc.)
- No replies ever sent to this sender
- High volume (>20 emails) with 0% reply rate

### Service Importance Score (0.0 to 1.0)

**Problem:** Not all service emails are equal. A Chase fraud alert is critical; a Chase marketing email is not.

**Solution:** `service_importance` distinguishes important service emails from noise.

```python
# Subject keywords that indicate importance (0.4 weight)
IMPORTANT_SUBJECT_KEYWORDS = [
    'order', 'shipped', 'delivered', 'payment', 'transaction',
    'confirm', 'receipt', 'invoice', 'alert', 'security',
    'verification', 'password', 'login', 'suspicious'
]

# Subject keywords that indicate low importance (0.4 weight)
LOW_IMPORTANCE_KEYWORDS = [
    'offer', 'sale', 'deal', 'save', 'discount', 'promo',
    'newsletter', 'digest', 'weekly', 'daily', 'tips'
]

# Sender address patterns that indicate importance (0.3 weight)
IMPORTANT_SENDER_PATTERNS = [
    'order', 'shipping', 'tracking', 'confirm', 'alert',
    'security', 'verify', 'transaction', 'payment'
]

service_importance = (
    0.4 * subject_importance_score +      # Subject keywords
    0.3 * sender_pattern_score +          # Sender address patterns
    0.3 * (1 if user_starred_similar else 0)  # User behavior signal
)
```

**Examples:**
| Email | service_importance |
|-------|-------------------|
| Chase: "Suspicious activity on your card" | 0.9 (alert + security) |
| Chase: "Your statement is ready" | 0.3 (no important keywords) |
| Amazon: "Your order has shipped" | 0.8 (shipped + order) |
| Amazon: "Deals just for you" | 0.1 (deal = low importance) |

**Query: Important service emails only**
```sql
SELECT e.subject, e.from_email, ef.service_importance
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.is_service_email = TRUE
  AND ef.service_importance > 0.5
ORDER BY ef.service_importance DESC;
```

### Urgency Score (0.0 to 1.0)

```python
urgency = (
    0.4 * relationship_strength +      # Important sender
    0.3 * recency_score +              # Recent email
    0.2 * (1 if is_business_hours else 0.5) +
    0.1 * (1 if has_attachments else 0)
)
```

## Implementation Plan

### Script: `scripts/compute_features.py`

1. **Read from database**
   - All emails from `emails` table
   - All users from `users` table (for relationship stats)

2. **Compute features in batches** (1000 emails at a time)
   - Relationship: Query user stats, compute metrics
   - Service: Check headers, body patterns, sender domain
   - Temporal: Parse date, extract time features
   - Content: Count words, attachments, recipients

3. **Insert into `email_features` table**
   - Batch insert for performance
   - Add index on `email_id` for fast joins

4. **Verify results**
   - Check feature distributions
   - Ensure no nulls where unexpected
   - Sample spot checks

### Performance

- Expected runtime: ~2-3 minutes for 25,596 emails
- No external API calls (all local computation)
- No cost

## What You Can Do After Phase 2

**Immediate queries:**
- Rank emails by importance
- Filter out service emails
- Find neglected relationships
- Analyze response patterns

**Enables Phase 3 & 4:**
- Phase 3 (embeddings) needs these features for hybrid ranking
- Phase 4 (LLM) can use these to filter what needs LLM analysis

**Future ML training:**
- These 30 features become input to ML model
- Train: "Given these features, predict if I'll reply"
- Result: Automatic email prioritization

## Files Created

1. `scripts/compute_features.py` - Feature computation script
2. `email_features` table in database
3. Update `docs/v2/ENRICHMENT_PIPELINE.md` with Phase 2 details

## Verification Queries

After running Phase 2:

```sql
-- Check coverage
SELECT COUNT(*) as total_emails,
       COUNT(ef.id) as with_features,
       ROUND(COUNT(ef.id) * 100.0 / COUNT(*), 1) as coverage_pct
FROM emails e
LEFT JOIN email_features ef ON ef.email_id = e.id
WHERE e.is_sent = FALSE;

-- Feature distribution
SELECT
    AVG(relationship_strength) as avg_relationship,
    AVG(urgency_score) as avg_urgency,
    SUM(CASE WHEN is_service_email THEN 1 ELSE 0 END) as service_count,
    COUNT(DISTINCT service_type) as service_types
FROM email_features;

-- Top relationships
SELECT e.from_email,
       ef.relationship_strength,
       ef.emails_from_sender_all,
       ef.user_replied_to_sender_rate
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE ef.relationship_strength > 0.5
GROUP BY e.from_email, ef.relationship_strength,
         ef.emails_from_sender_all, ef.user_replied_to_sender_rate
ORDER BY ef.relationship_strength DESC
LIMIT 10;
```

## Dependencies

- ✅ Phase 1 complete (action labels computed)
- ✅ `users` table populated (sender stats)
- ✅ `emails` table has headers, body, labels

## Timeline Estimate

- Script development: Already exists in docs
- Testing: 1 session
- Full computation: 2-3 minutes
- Verification: Quick queries
- **Total: Can complete in this session**

---

**Ready to proceed?** Once approved, I'll:
1. Create `compute_features.py` script
2. Run it on gmail_twoyrs database
3. Verify results with queries
4. Commit and push to rl-emails/main
5. Close Phase 2 bead
