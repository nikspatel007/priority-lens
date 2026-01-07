"""Shared test fixtures for rl-emails."""
from __future__ import annotations

import os
from typing import Any, Generator

import pytest


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Test database URL (separate from prod).

    Uses TEST_DATABASE_URL env var if set, otherwise uses a default test database.
    """
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5433/test_rl_emails"
    )


@pytest.fixture
def sample_email() -> dict[str, Any]:
    """Sample email data for testing."""
    return {
        "message_id": "<test123@example.com>",
        "from_email": "sender@example.com",
        "from_name": "Test Sender",
        "to_emails": ["recipient@example.com"],
        "cc_emails": [],
        "bcc_emails": [],
        "subject": "Test Subject",
        "body_text": "This is a test email body with enough content to be meaningful.",
        "body_html": "<html><body>This is a test email body.</body></html>",
        "date_str": "Mon, 1 Jan 2024 10:00:00 +0000",
        "headers": {"X-Test": "true"},
        "labels": ["INBOX"],
        "in_reply_to": None,
        "references": [],
    }


@pytest.fixture
def sample_email_with_reply() -> dict[str, Any]:
    """Sample email that is a reply."""
    return {
        "message_id": "<reply456@example.com>",
        "from_email": "recipient@example.com",
        "from_name": "Test Recipient",
        "to_emails": ["sender@example.com"],
        "cc_emails": [],
        "bcc_emails": [],
        "subject": "Re: Test Subject",
        "body_text": "This is a reply to the test email.",
        "body_html": None,
        "date_str": "Mon, 1 Jan 2024 11:00:00 +0000",
        "headers": {},
        "labels": ["SENT"],
        "in_reply_to": "<test123@example.com>",
        "references": ["<test123@example.com>"],
    }


@pytest.fixture
def sample_service_email() -> dict[str, Any]:
    """Sample service/automated email."""
    return {
        "message_id": "<noreply789@service.example.com>",
        "from_email": "noreply@service.example.com",
        "from_name": None,
        "to_emails": ["user@example.com"],
        "cc_emails": [],
        "bcc_emails": [],
        "subject": "Your order has shipped",
        "body_text": "Your order #12345 has been shipped. Track at: http://track.example.com",
        "body_html": None,
        "date_str": "Mon, 1 Jan 2024 12:00:00 +0000",
        "headers": {"List-Unsubscribe": "<mailto:unsub@service.example.com>"},
        "labels": ["INBOX"],
        "in_reply_to": None,
        "references": [],
    }


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
    monkeypatch.setenv("MBOX_PATH", "/tmp/test.mbox")
    monkeypatch.setenv("YOUR_EMAIL", "test@example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    yield
