-- Stage 6: Database Schema for rl-emails
-- Creates tables per docs/v2/SCHEMA.md

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Drop existing tables if rebuilding
DROP TABLE IF EXISTS attachments CASCADE;
DROP TABLE IF EXISTS threads CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS emails CASCADE;
DROP TABLE IF EXISTS raw_emails CASCADE;

-- ============================================
-- raw_emails (Source of Truth)
-- Immutable raw data from MBOX parse
-- ============================================
CREATE TABLE raw_emails (
    id SERIAL PRIMARY KEY,
    message_id TEXT UNIQUE NOT NULL,

    -- Raw headers (exactly as parsed from MBOX)
    thread_id TEXT,
    in_reply_to TEXT,
    references_raw TEXT,

    date_raw TEXT,
    from_raw TEXT,
    to_raw TEXT,
    cc_raw TEXT,
    bcc_raw TEXT,
    subject_raw TEXT,

    -- Raw content
    body_text TEXT,
    body_html TEXT,

    -- Gmail metadata
    labels_raw TEXT,

    -- MBOX metadata
    mbox_offset BIGINT,
    raw_size_bytes INTEGER,

    imported_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_raw_emails_message_id ON raw_emails(message_id);
CREATE INDEX idx_raw_emails_thread_id ON raw_emails(thread_id);

-- ============================================
-- emails (Enriched/Derived)
-- Derived from raw_emails with parsed fields
-- ============================================
CREATE TABLE emails (
    id SERIAL PRIMARY KEY,
    raw_email_id INTEGER REFERENCES raw_emails(id),
    message_id TEXT UNIQUE NOT NULL,
    thread_id TEXT,
    in_reply_to TEXT,

    -- Parsed headers
    date_parsed TIMESTAMPTZ,
    from_email TEXT,
    from_name TEXT,
    to_emails TEXT[],
    cc_emails TEXT[],
    subject TEXT,

    -- Content
    body_text TEXT,
    body_preview TEXT,
    word_count INTEGER,

    -- Parsed Gmail metadata
    labels TEXT[],

    -- Attachment summary
    has_attachments BOOLEAN DEFAULT FALSE,
    attachment_count INTEGER DEFAULT 0,
    attachment_types TEXT[],
    total_attachment_bytes BIGINT DEFAULT 0,

    -- Ownership
    is_sent BOOLEAN DEFAULT FALSE,

    -- Computed enrichment (Stage 5)
    action TEXT,
    timing TEXT,
    response_time_seconds INTEGER,
    priority_score FLOAT,

    -- Processing metadata
    enriched_at TIMESTAMPTZ,
    enrichment_version INTEGER DEFAULT 1,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_emails_thread_id ON emails(thread_id);
CREATE INDEX idx_emails_from_email ON emails(from_email);
CREATE INDEX idx_emails_date ON emails(date_parsed);
CREATE INDEX idx_emails_in_reply_to ON emails(in_reply_to);
CREATE INDEX idx_emails_is_sent ON emails(is_sent);
CREATE INDEX idx_emails_labels ON emails USING GIN(labels);
CREATE INDEX idx_emails_subject_trgm ON emails USING GIN(subject gin_trgm_ops);

-- ============================================
-- attachments
-- Individual attachment metadata
-- ============================================
CREATE TABLE attachments (
    id SERIAL PRIMARY KEY,
    raw_email_id INTEGER REFERENCES raw_emails(id),
    email_id INTEGER REFERENCES emails(id),

    -- Attachment metadata
    filename TEXT,
    content_type TEXT,
    size_bytes INTEGER,
    content_disposition TEXT,

    -- Hashing for dedup
    content_hash TEXT,

    -- Storage
    stored_path TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_attachments_raw_email_id ON attachments(raw_email_id);
CREATE INDEX idx_attachments_email_id ON attachments(email_id);
CREATE INDEX idx_attachments_content_type ON attachments(content_type);

-- ============================================
-- threads (Aggregated)
-- Thread-level aggregations
-- ============================================
CREATE TABLE threads (
    id SERIAL PRIMARY KEY,
    thread_id TEXT UNIQUE NOT NULL,
    subject TEXT,

    -- Participants
    participants TEXT[],
    your_role TEXT,

    -- Email stats
    email_count INTEGER DEFAULT 0,
    your_email_count INTEGER DEFAULT 0,
    your_reply_count INTEGER DEFAULT 0,

    -- Attachment stats
    has_attachments BOOLEAN DEFAULT FALSE,
    total_attachment_count INTEGER DEFAULT 0,
    attachment_types TEXT[],

    -- Timing
    started_at TIMESTAMPTZ,
    last_activity TIMESTAMPTZ,
    your_first_reply_at TIMESTAMPTZ,

    -- Computed
    avg_response_time_seconds INTEGER,
    thread_duration_seconds INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_threads_thread_id ON threads(thread_id);
CREATE INDEX idx_threads_has_attachments ON threads(has_attachments);

-- ============================================
-- users (Aggregated)
-- Email addresses with communication stats
-- ============================================
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,

    -- Relationship
    is_you BOOLEAN DEFAULT FALSE,

    -- Communication stats
    emails_from INTEGER DEFAULT 0,
    emails_to INTEGER DEFAULT 0,
    threads_with INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,

    -- Response patterns
    avg_response_time_seconds INTEGER,
    reply_rate FLOAT,

    -- Labels
    is_important_sender BOOLEAN DEFAULT FALSE,
    first_contact TIMESTAMPTZ,
    last_contact TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_you ON users(is_you);

-- ============================================
-- Summary
-- ============================================
-- Tables created:
--   raw_emails: Immutable source data
--   emails: Enriched/derived data
--   attachments: Attachment metadata
--   threads: Thread-level aggregations
--   users: Communication stats per user
