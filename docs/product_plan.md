# AI Email Agent - Product Plan

## Vision

An AI agent that manages your inbox by taking real actions on your behalf:
- Automatically files newsletters, notifications, and FYI emails
- Drafts replies for your review (never auto-sends)
- Learns from your feedback to improve over time

## Current State (Completed)

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Action labels (REPLIED, ARCHIVED, etc.) | Done |
| Phase 2 | ML features (relationship, urgency, service detection) | Done |
| Phase 3 | Semantic embeddings (22,618 emails) | Done |
| Phase 0 | Rule-based AI classification | Done |
| Phase 4 | LLM classification (7,474 needs_llm emails) | Done |

### Classification Results
- **ai_full**: 9,876 (44%) - Can auto-handle
- **needs_llm**: 7,614 (34%) - LLM classified
- **ai_partial**: 2,680 (12%) - AI drafts, human approves
- **human_required**: 2,448 (11%) - Human decision needed

---

## Product Roadmap

### Phase 5: Gmail Integration & Execution Engine

**Goal**: Connect to Gmail and execute actions

#### 5.1 Gmail OAuth & Client
- OAuth 2.0 authentication flow
- Gmail API wrapper (read, label, archive, draft)
- Polling-based sync (5-minute intervals)

#### 5.2 Action Execution
- Map classifications to Gmail actions:
  - `ai_full` → auto-archive or apply label
  - `ai_partial` → create draft for review
  - `human_required` → surface in approval queue
- Confidence gateway routes low-confidence to approval
- Full audit trail in database

#### 5.3 Safety Controls
- Never auto-send (drafts only)
- Conservative confidence thresholds
- Rollback capability for labels/archive

### Phase 6: Approval & Dashboard UI

**Goal**: Human review interface

- Approval queue for pending actions
- Email preview + predicted action + confidence
- Batch approve/reject
- Action history and audit log

### Phase 7: Feedback Loop & Online Learning

**Goal**: Learn from execution outcomes

#### 7.1 Outcome Tracking
- Did user send the draft?
- Did user edit the draft significantly?
- Did user move archived email back to inbox?
- Did user change applied labels?

#### 7.2 Reward Signals
| Outcome | Reward |
|---------|--------|
| Draft sent unchanged | +1.0 |
| Draft sent with edits | +0.5 - edit_penalty |
| Draft deleted | -0.5 |
| Archive stayed | +0.5 |
| User undid action | -1.0 |

#### 7.3 Online Learning
- Aggregate feedback into training batches
- Update model weights weekly
- Track accuracy over time

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              APPROVAL UI (Streamlit)            │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              EXECUTION ENGINE                   │
│   ActionMapper │ ConfidenceGateway │ Executor   │
└────────┬────────────────────────────┬───────────┘
         │                            │
┌────────▼────────┐      ┌────────────▼───────────┐
│   GMAIL LAYER   │      │   CLASSIFICATION       │
│  OAuth + Client │      │  Rules + LLM + ML      │
└────────┬────────┘      └────────────────────────┘
         │
┌────────▼────────────────────────────────────────┐
│              FEEDBACK LOOP                      │
│  OutcomeTracker │ RewardComputer │ Learner      │
└─────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           PostgreSQL + pgvector                 │
└─────────────────────────────────────────────────┘
```

---

## Database Schema Additions

### gmail_credentials
```sql
CREATE TABLE gmail_credentials (
    id SERIAL PRIMARY KEY,
    user_email TEXT UNIQUE NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    token_expiry TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### agent_actions
```sql
CREATE TABLE agent_actions (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id),
    gmail_message_id TEXT,
    predicted_action TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    action_payload JSONB,
    reasoning TEXT,
    status TEXT DEFAULT 'pending',
    requires_approval BOOLEAN DEFAULT FALSE,
    executed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### action_outcomes
```sql
CREATE TABLE action_outcomes (
    id SERIAL PRIMARY KEY,
    action_id INTEGER REFERENCES agent_actions(id),
    draft_sent BOOLEAN,
    draft_edited BOOLEAN,
    draft_deleted BOOLEAN,
    user_moved_back BOOLEAN,
    reward_signal FLOAT,
    outcome_observed_at TIMESTAMPTZ
);
```

---

## File Structure

```
rl-emails/
├── src/
│   ├── gmail/
│   │   ├── oauth.py          # OAuth 2.0 flow
│   │   ├── client.py         # Gmail API wrapper
│   │   └── sync.py           # Polling sync
│   ├── agent/
│   │   ├── executor.py       # Action execution
│   │   ├── action_mapper.py  # Classification → action
│   │   ├── confidence_gateway.py
│   │   └── draft_generator.py
│   └── feedback/
│       ├── outcome_tracker.py
│       ├── reward_computer.py
│       └── online_learner.py
├── apps/
│   ├── agent_dashboard.py
│   └── approval_queue.py
└── scripts/
    ├── gmail_auth.py
    └── run_agent.py
```

---

## Confidence Thresholds

| Action | Auto-execute | Needs Approval |
|--------|--------------|----------------|
| archive | >= 0.90 | < 0.90 |
| apply_label | >= 0.85 | < 0.85 |
| create_draft | >= 0.70 | < 0.70 |

Start conservative, tune based on approval rates and user feedback.

---

## Success Metrics

1. **Automation Rate**: % of emails handled without human intervention
2. **Accuracy**: % of auto-actions not undone by user
3. **Draft Quality**: % of drafts sent without significant edits
4. **Time Saved**: Estimated hours saved per week
5. **Learning Curve**: Accuracy improvement over time

---

## MVP Scope

**First Release**: Gmail OAuth + auto-archive for `ai_full` emails
1. Connect to Gmail
2. Identify newsletter/notification emails
3. Auto-archive with confidence > 0.90
4. Log all actions for audit
5. Manual review of first 100 actions before expanding

---

## Dependencies

```
google-auth
google-auth-oauthlib
google-api-python-client
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Wrong email archived | Rollback capability, conservative thresholds |
| Bad draft sent | Never auto-send, drafts only |
| API rate limits | Respect Gmail quotas, backoff |
| Token expiry | Auto-refresh, graceful degradation |
| Model drift | Weekly retraining, accuracy monitoring |
