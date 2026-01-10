"""Tests for Gmail API models."""

from __future__ import annotations

from datetime import UTC, datetime

from priority_lens.integrations.gmail.models import GmailMessage, GmailMessageRef


class TestGmailMessageRef:
    """Tests for GmailMessageRef dataclass."""

    def test_create_message_ref(self) -> None:
        """Test creating a message reference."""
        ref = GmailMessageRef(id="msg123", thread_id="thread456")

        assert ref.id == "msg123"
        assert ref.thread_id == "thread456"

    def test_message_ref_equality(self) -> None:
        """Test message ref equality."""
        ref1 = GmailMessageRef(id="msg123", thread_id="thread456")
        ref2 = GmailMessageRef(id="msg123", thread_id="thread456")

        assert ref1 == ref2

    def test_message_ref_inequality(self) -> None:
        """Test message ref inequality."""
        ref1 = GmailMessageRef(id="msg123", thread_id="thread456")
        ref2 = GmailMessageRef(id="msg789", thread_id="thread456")

        assert ref1 != ref2


class TestGmailMessage:
    """Tests for GmailMessage dataclass."""

    def test_create_message(self) -> None:
        """Test creating a full Gmail message."""
        now = datetime.now(UTC)
        msg = GmailMessage(
            id="msg123",
            thread_id="thread456",
            history_id="12345",
            message_id="<test@example.com>",
            subject="Test Subject",
            from_address="sender@example.com",
            date_sent=now,
            snippet="This is a preview...",
            size_bytes=1024,
            has_attachments=False,
            label_ids=["INBOX", "UNREAD"],
            to_addresses=["recipient@example.com"],
            cc_addresses=["cc@example.com"],
            body_plain="This is the body",
            body_html="<p>This is the body</p>",
        )

        assert msg.id == "msg123"
        assert msg.thread_id == "thread456"
        assert msg.history_id == "12345"
        assert "INBOX" in msg.label_ids
        assert msg.subject == "Test Subject"
        assert msg.size_bytes == 1024
        assert msg.has_attachments is False

    def test_is_sent_true(self) -> None:
        """Test is_sent returns True when SENT label present."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="me@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["SENT"],
            to_addresses=["you@example.com"],
        )

        assert msg.is_sent() is True

    def test_is_sent_false(self) -> None:
        """Test is_sent returns False when SENT label not present."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["INBOX"],
            to_addresses=["me@example.com"],
        )

        assert msg.is_sent() is False

    def test_is_inbox(self) -> None:
        """Test is_inbox method."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["INBOX", "UNREAD"],
            to_addresses=["me@example.com"],
        )

        assert msg.is_inbox() is True

    def test_is_unread(self) -> None:
        """Test is_unread method."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["INBOX", "UNREAD"],
            to_addresses=["me@example.com"],
        )

        assert msg.is_unread() is True

    def test_is_starred(self) -> None:
        """Test is_starred method."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["STARRED"],
            to_addresses=["me@example.com"],
        )

        assert msg.is_starred() is True

    def test_is_draft(self) -> None:
        """Test is_draft method."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="me@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["DRAFT"],
        )

        assert msg.is_draft() is True

    def test_is_spam(self) -> None:
        """Test is_spam method."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Win big!",
            from_address="spam@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["SPAM"],
            to_addresses=["me@example.com"],
        )

        assert msg.is_spam() is True

    def test_is_trash(self) -> None:
        """Test is_trash method."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            label_ids=["TRASH"],
            to_addresses=["me@example.com"],
        )

        assert msg.is_trash() is True

    def test_default_lists(self) -> None:
        """Test that list fields have default empty lists."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
        )

        assert msg.to_addresses == []
        assert msg.cc_addresses == []
        assert msg.label_ids == []
        assert msg.references == []
        assert msg.in_reply_to is None
        assert msg.body_plain is None
        assert msg.body_html is None

    def test_message_with_reply_headers(self) -> None:
        """Test message with reply headers."""
        msg = GmailMessage(
            id="msg2",
            thread_id="t1",
            history_id="h2",
            message_id="<m2@example.com>",
            subject="Re: Test",
            from_address="me@example.com",
            date_sent=datetime.now(UTC),
            snippet="Thanks for...",
            size_bytes=200,
            has_attachments=False,
            label_ids=["SENT"],
            to_addresses=["other@example.com"],
            in_reply_to="<m1@example.com>",
            references=["<m1@example.com>"],
            body_plain="Thanks for the email",
        )

        assert msg.in_reply_to == "<m1@example.com>"
        assert "<m1@example.com>" in msg.references

    def test_message_with_attachments(self) -> None:
        """Test message with attachments."""
        msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<m1>",
            subject="Test with attachment",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="See attached",
            size_bytes=1024000,
            has_attachments=True,
            label_ids=["INBOX"],
            to_addresses=["me@example.com"],
            body_plain="See attached file",
        )

        assert msg.has_attachments is True
        assert msg.size_bytes == 1024000
