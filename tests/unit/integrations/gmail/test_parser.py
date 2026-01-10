"""Tests for Gmail message parser."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from priority_lens.integrations.gmail.models import GmailMessage
from priority_lens.integrations.gmail.parser import (
    GmailParseError,
    _decode_base64,
    _has_attachments,
    _parse_body,
    _parse_date,
    _parse_headers,
    _parse_references,
    _split_addresses,
    gmail_to_email_data,
    parse_email_address,
    parse_raw_message,
)


class TestGmailParseError:
    """Tests for GmailParseError exception."""

    def test_create_error_with_all_fields(self) -> None:
        """Test creating error with all fields."""
        error = GmailParseError(
            reason="Missing required field",
            message_id="msg123",
            field="subject",
        )

        assert str(error) == "Missing required field"
        assert error.reason == "Missing required field"
        assert error.message_id == "msg123"
        assert error.field == "subject"

    def test_create_error_minimal(self) -> None:
        """Test creating error with only reason."""
        error = GmailParseError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.reason == "Something went wrong"
        assert error.message_id is None
        assert error.field is None


class TestParseEmailAddress:
    """Tests for parse_email_address function."""

    def test_name_and_email_format(self) -> None:
        """Test 'Name <email>' format."""
        email, name = parse_email_address("John Doe <john@example.com>")

        assert email == "john@example.com"
        assert name == "John Doe"

    def test_quoted_name_format(self) -> None:
        """Test quoted name format."""
        email, name = parse_email_address('"Jane Doe" <jane@example.com>')

        assert email == "jane@example.com"
        assert name == "Jane Doe"

    def test_email_in_brackets_only(self) -> None:
        """Test '<email>' format without name."""
        email, name = parse_email_address("<noreply@example.com>")

        assert email == "noreply@example.com"
        assert name is None

    def test_bare_email(self) -> None:
        """Test bare email address."""
        email, name = parse_email_address("user@example.com")

        assert email == "user@example.com"
        assert name is None

    def test_email_lowercase(self) -> None:
        """Test that email is lowercased."""
        email, name = parse_email_address("USER@EXAMPLE.COM")

        assert email == "user@example.com"

    def test_empty_string(self) -> None:
        """Test empty string input."""
        email, name = parse_email_address("")

        assert email == ""
        assert name is None

    def test_no_at_sign(self) -> None:
        """Test string without @ sign."""
        email, name = parse_email_address("not-an-email")

        assert email == "not-an-email"
        assert name is None

    def test_whitespace_trimmed(self) -> None:
        """Test that whitespace is trimmed."""
        email, name = parse_email_address("  user@example.com  ")

        assert email == "user@example.com"

    def test_empty_name_in_brackets(self) -> None:
        """Test empty name with email in brackets."""
        email, name = parse_email_address(" <email@example.com>")

        assert email == "email@example.com"
        assert name is None


class TestParseRawMessage:
    """Tests for parse_raw_message function."""

    def test_parses_basic_message(self) -> None:
        """Test parsing a basic Gmail API response."""
        raw = {
            "id": "msg123",
            "threadId": "thread456",
            "historyId": "789",
            "labelIds": ["INBOX", "UNREAD"],
            "snippet": "Preview text...",
            "sizeEstimate": 1024,
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    {"name": "Message-ID", "value": "<msg@example.com>"},
                ],
                "mimeType": "text/plain",
                "body": {"data": _encode_base64("Hello World")},
            },
        }

        msg = parse_raw_message(raw)

        assert msg.id == "msg123"
        assert msg.thread_id == "thread456"
        assert msg.history_id == "789"
        assert msg.subject == "Test Subject"
        assert msg.from_address == "sender@example.com"
        assert "recipient@example.com" in msg.to_addresses
        assert msg.body_plain == "Hello World"
        assert msg.label_ids == ["INBOX", "UNREAD"]
        assert msg.snippet == "Preview text..."
        assert msg.size_bytes == 1024

    def test_missing_message_id_raises_error(self) -> None:
        """Test that missing message ID raises error."""
        raw: dict[str, object] = {"threadId": "thread1"}

        with pytest.raises(GmailParseError, match="Missing message ID"):
            parse_raw_message(raw)

    def test_generates_message_id_if_missing(self) -> None:
        """Test that Message-ID header is generated if missing."""
        raw = {
            "id": "gmail123",
            "threadId": "thread1",
            "payload": {"headers": []},
        }

        msg = parse_raw_message(raw)

        assert msg.message_id == "<gmail123@gmail.com>"

    def test_handles_missing_payload(self) -> None:
        """Test handling missing payload gracefully."""
        raw: dict[str, object] = {
            "id": "msg1",
            "threadId": "thread1",
        }

        msg = parse_raw_message(raw)

        assert msg.id == "msg1"
        assert msg.subject == ""
        assert msg.body_plain is None

    def test_handles_invalid_payload_type(self) -> None:
        """Test handling non-dict payload."""
        raw: dict[str, object] = {
            "id": "msg1",
            "threadId": "thread1",
            "payload": "invalid",
        }

        msg = parse_raw_message(raw)

        assert msg.id == "msg1"

    def test_handles_non_list_label_ids(self) -> None:
        """Test handling non-list labelIds."""
        raw: dict[str, object] = {
            "id": "msg1",
            "threadId": "thread1",
            "labelIds": "INBOX",
        }

        msg = parse_raw_message(raw)

        assert msg.label_ids == []

    def test_parses_cc_addresses(self) -> None:
        """Test parsing CC addresses."""
        raw = {
            "id": "msg1",
            "threadId": "thread1",
            "payload": {
                "headers": [
                    {"name": "Cc", "value": "cc1@example.com, cc2@example.com"},
                ],
            },
        }

        msg = parse_raw_message(raw)

        assert "cc1@example.com" in msg.cc_addresses
        assert "cc2@example.com" in msg.cc_addresses

    def test_parses_reply_headers(self) -> None:
        """Test parsing In-Reply-To and References headers."""
        raw = {
            "id": "msg1",
            "threadId": "thread1",
            "payload": {
                "headers": [
                    {"name": "In-Reply-To", "value": "<original@example.com>"},
                    {
                        "name": "References",
                        "value": "<first@example.com> <second@example.com>",
                    },
                ],
            },
        }

        msg = parse_raw_message(raw)

        assert msg.in_reply_to == "<original@example.com>"
        assert "<first@example.com>" in msg.references
        assert "<second@example.com>" in msg.references

    def test_parses_multipart_message(self) -> None:
        """Test parsing multipart message with parts."""
        raw = {
            "id": "msg1",
            "threadId": "thread1",
            "payload": {
                "mimeType": "multipart/alternative",
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _encode_base64("Plain text")},
                    },
                    {
                        "mimeType": "text/html",
                        "body": {"data": _encode_base64("<p>HTML</p>")},
                    },
                ],
            },
        }

        msg = parse_raw_message(raw)

        assert msg.body_plain == "Plain text"
        assert msg.body_html == "<p>HTML</p>"

    def test_detects_attachments(self) -> None:
        """Test detecting attachments in message."""
        raw = {
            "id": "msg1",
            "threadId": "thread1",
            "payload": {
                "mimeType": "multipart/mixed",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": ""}},
                    {
                        "mimeType": "application/pdf",
                        "filename": "document.pdf",
                    },
                ],
            },
        }

        msg = parse_raw_message(raw)

        assert msg.has_attachments is True


class TestGmailToEmailData:
    """Tests for gmail_to_email_data function."""

    def test_converts_basic_message(self) -> None:
        """Test converting basic GmailMessage to EmailData."""
        now = datetime.now(UTC)
        gmail_msg = GmailMessage(
            id="gmail123",
            thread_id="thread456",
            history_id="hist789",
            message_id="<msg@example.com>",
            subject="Test Subject",
            from_address="John Doe <john@example.com>",
            date_sent=now,
            snippet="Preview...",
            size_bytes=1024,
            has_attachments=False,
            label_ids=["INBOX"],
            to_addresses=["recipient@example.com"],
            cc_addresses=["cc@example.com"],
            body_plain="Hello World",
            body_html="<p>Hello World</p>",
        )

        email_data = gmail_to_email_data(gmail_msg)

        assert email_data["message_id"] == "<msg@example.com>"
        assert email_data["from_email"] == "john@example.com"
        assert email_data["from_name"] == "John Doe"
        assert email_data["subject"] == "Test Subject"
        assert email_data["body_text"] == "Hello World"
        assert email_data["body_html"] == "<p>Hello World</p>"
        assert email_data["labels"] == ["INBOX"]
        assert "recipient@example.com" in email_data["to_emails"]
        assert "cc@example.com" in email_data["cc_emails"]

    def test_handles_none_body(self) -> None:
        """Test handling None body fields."""
        gmail_msg = GmailMessage(
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
            body_plain=None,
            body_html=None,
        )

        email_data = gmail_to_email_data(gmail_msg)

        assert email_data["body_text"] == ""
        assert email_data["body_html"] is None

    def test_builds_headers_dict(self) -> None:
        """Test that headers dict is built correctly."""
        gmail_msg = GmailMessage(
            id="msg1",
            thread_id="t1",
            history_id="h1",
            message_id="<msg@example.com>",
            subject="Test",
            from_address="sender@example.com",
            date_sent=datetime.now(UTC),
            snippet="",
            size_bytes=100,
            has_attachments=False,
            to_addresses=["to@example.com"],
            cc_addresses=["cc@example.com"],
            in_reply_to="<reply@example.com>",
            references=["<ref1@example.com>", "<ref2@example.com>"],
        )

        email_data = gmail_to_email_data(gmail_msg)

        headers = email_data["headers"]
        assert headers["message-id"] == "<msg@example.com>"
        assert headers["from"] == "sender@example.com"
        assert headers["to"] == "to@example.com"
        assert headers["cc"] == "cc@example.com"
        assert headers["in-reply-to"] == "<reply@example.com>"
        assert "<ref1@example.com>" in headers["references"]

    def test_empty_bcc_list(self) -> None:
        """Test that bcc_emails is always empty list."""
        gmail_msg = GmailMessage(
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

        email_data = gmail_to_email_data(gmail_msg)

        assert email_data["bcc_emails"] == []


class TestParseHeaders:
    """Tests for _parse_headers helper."""

    def test_parses_headers_to_lowercase(self) -> None:
        """Test that header names are lowercased."""
        payload = {
            "headers": [
                {"name": "Subject", "value": "Test"},
                {"name": "FROM", "value": "sender@example.com"},
            ]
        }

        headers = _parse_headers(payload)

        assert headers["subject"] == "Test"
        assert headers["from"] == "sender@example.com"

    def test_handles_non_list_headers(self) -> None:
        """Test handling non-list headers value."""
        payload: dict[str, object] = {"headers": "invalid"}

        headers = _parse_headers(payload)

        assert headers == {}

    def test_handles_non_dict_header_entries(self) -> None:
        """Test handling non-dict entries in headers list."""
        payload: dict[str, object] = {"headers": ["invalid", {"name": "Subject", "value": "Test"}]}

        headers = _parse_headers(payload)

        assert headers["subject"] == "Test"

    def test_skips_empty_header_names(self) -> None:
        """Test that empty header names are skipped."""
        payload: dict[str, object] = {"headers": [{"name": "", "value": "value"}]}

        headers = _parse_headers(payload)

        assert "" not in headers


class TestSplitAddresses:
    """Tests for _split_addresses helper."""

    def test_splits_comma_separated(self) -> None:
        """Test splitting comma-separated addresses."""
        result = _split_addresses("a@example.com, b@example.com")

        assert len(result) == 2
        assert "a@example.com" in result
        assert "b@example.com" in result

    def test_handles_empty_string(self) -> None:
        """Test handling empty string."""
        result = _split_addresses("")

        assert result == []

    def test_handles_quoted_names_with_commas(self) -> None:
        """Test handling quoted names that contain commas."""
        result = _split_addresses('"Doe, John" <john@example.com>, jane@example.com')

        assert len(result) == 2
        assert "john@example.com" in result
        assert "jane@example.com" in result

    def test_skips_empty_parts(self) -> None:
        """Test that empty parts after split are skipped."""
        result = _split_addresses("a@example.com, , , b@example.com")

        assert len(result) == 2
        assert "a@example.com" in result
        assert "b@example.com" in result

    def test_skips_parts_without_valid_email(self) -> None:
        """Test that empty parts after stripping don't result in empty emails."""
        # Test with parts that become empty after strip
        result = _split_addresses("valid@example.com,   , other@example.com")

        assert len(result) == 2
        assert "valid@example.com" in result
        assert "other@example.com" in result

    def test_skips_empty_email_from_parse(self) -> None:
        """Test that parsed empty emails are skipped."""
        # <  > (spaces in brackets) parses to empty email
        result = _split_addresses("<  >, valid@example.com")

        assert len(result) == 1
        assert "valid@example.com" in result


class TestParseReferences:
    """Tests for _parse_references helper."""

    def test_parses_space_separated_refs(self) -> None:
        """Test parsing space-separated references."""
        result = _parse_references("<ref1@example.com> <ref2@example.com>")

        assert len(result) == 2
        assert "<ref1@example.com>" in result
        assert "<ref2@example.com>" in result

    def test_handles_empty_string(self) -> None:
        """Test handling empty string."""
        result = _parse_references("")

        assert result == []

    def test_adds_brackets_to_bare_refs(self) -> None:
        """Test adding brackets to bare message IDs."""
        result = _parse_references("ref@example.com")

        assert result == ["<ref@example.com>"]

    def test_skips_invalid_refs(self) -> None:
        """Test that refs without @ or brackets are skipped."""
        result = _parse_references("<valid@example.com> invalid-no-at <another@example.com>")

        assert len(result) == 2
        assert "<valid@example.com>" in result
        assert "<another@example.com>" in result
        assert "invalid-no-at" not in result


class TestParseDate:
    """Tests for _parse_date helper."""

    def test_parses_rfc_2822_format(self) -> None:
        """Test parsing RFC 2822 date format."""
        result = _parse_date("Mon, 1 Jan 2024 12:00:00 +0000")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parses_iso_format(self) -> None:
        """Test parsing ISO format with Z suffix."""
        result = _parse_date("2024-01-01T12:00:00Z")

        assert result.year == 2024
        assert result.month == 1

    def test_returns_current_time_for_empty(self) -> None:
        """Test returning current time for empty string."""
        before = datetime.now(UTC)
        result = _parse_date("")
        after = datetime.now(UTC)

        assert before <= result <= after

    def test_returns_current_time_for_invalid(self) -> None:
        """Test returning current time for invalid date."""
        before = datetime.now(UTC)
        result = _parse_date("not-a-date")
        after = datetime.now(UTC)

        assert before <= result <= after


class TestParseBody:
    """Tests for _parse_body helper."""

    def test_parses_plain_text_body(self) -> None:
        """Test parsing plain text body."""
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _encode_base64("Hello World")},
        }

        plain, html = _parse_body(payload)

        assert plain == "Hello World"
        assert html is None

    def test_parses_html_body(self) -> None:
        """Test parsing HTML body."""
        payload = {
            "mimeType": "text/html",
            "body": {"data": _encode_base64("<p>Hello</p>")},
        }

        plain, html = _parse_body(payload)

        assert plain is None
        assert html == "<p>Hello</p>"

    def test_parses_multipart_body(self) -> None:
        """Test parsing multipart message."""
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _encode_base64("Plain")}},
                {"mimeType": "text/html", "body": {"data": _encode_base64("<p>HTML</p>")}},
            ],
        }

        plain, html = _parse_body(payload)

        assert plain == "Plain"
        assert html == "<p>HTML</p>"

    def test_handles_non_string_data(self) -> None:
        """Test handling non-string body data."""
        payload: dict[str, object] = {
            "mimeType": "text/plain",
            "body": {"data": 12345},
        }

        plain, html = _parse_body(payload)

        assert plain is None
        assert html is None

    def test_handles_non_dict_body(self) -> None:
        """Test handling non-dict body."""
        payload: dict[str, object] = {
            "mimeType": "text/plain",
            "body": "invalid",
        }

        plain, html = _parse_body(payload)

        assert plain is None
        assert html is None

    def test_handles_non_dict_parts(self) -> None:
        """Test handling non-dict entries in parts."""
        payload: dict[str, object] = {
            "mimeType": "multipart/mixed",
            "parts": [
                "invalid",
                {"mimeType": "text/plain", "body": {"data": _encode_base64("Valid")}},
            ],
        }

        plain, html = _parse_body(payload)

        assert plain == "Valid"

    def test_handles_non_list_parts(self) -> None:
        """Test handling non-list parts."""
        payload: dict[str, object] = {
            "mimeType": "multipart/mixed",
            "parts": "invalid",
        }

        plain, html = _parse_body(payload)

        assert plain is None
        assert html is None

    def test_ignores_unknown_mime_type(self) -> None:
        """Test that unknown mime types are ignored."""
        payload = {
            "mimeType": "application/json",
            "body": {"data": _encode_base64('{"key": "value"}')},
        }

        plain, html = _parse_body(payload)

        # Neither plain nor html should be set for unknown mime type
        assert plain is None
        assert html is None


class TestDecodeBase64:
    """Tests for _decode_base64 helper."""

    def test_decodes_base64url(self) -> None:
        """Test decoding base64url encoded string."""
        encoded = _encode_base64("Hello World")
        result = _decode_base64(encoded)

        assert result == "Hello World"

    def test_handles_padding(self) -> None:
        """Test handling strings needing padding."""
        # Encode without padding
        data = base64.urlsafe_b64encode(b"Hi").decode().rstrip("=")
        result = _decode_base64(data)

        assert result == "Hi"

    def test_handles_invalid_input(self) -> None:
        """Test handling invalid base64 input that causes decode error."""
        # Single character is invalid base64 (needs at least 2 chars after padding)
        result = _decode_base64("!")

        assert result == ""


class TestHasAttachments:
    """Tests for _has_attachments helper."""

    def test_detects_attachment_by_filename(self) -> None:
        """Test detecting attachment by filename."""
        payload = {
            "parts": [
                {"mimeType": "text/plain"},
                {"mimeType": "application/pdf", "filename": "doc.pdf"},
            ]
        }

        assert _has_attachments(payload) is True

    def test_detects_attachment_by_disposition(self) -> None:
        """Test detecting attachment by Content-Disposition header."""
        payload = {
            "parts": [
                {
                    "mimeType": "application/pdf",
                    "headers": [
                        {"name": "Content-Disposition", "value": "attachment; filename=doc.pdf"}
                    ],
                }
            ]
        }

        assert _has_attachments(payload) is True

    def test_no_attachments(self) -> None:
        """Test no attachments detected."""
        payload = {
            "parts": [
                {"mimeType": "text/plain"},
                {"mimeType": "text/html"},
            ]
        }

        assert _has_attachments(payload) is False

    def test_handles_non_list_parts(self) -> None:
        """Test handling non-list parts."""
        payload: dict[str, object] = {"parts": "invalid"}

        assert _has_attachments(payload) is False

    def test_handles_non_dict_part(self) -> None:
        """Test handling non-dict entries in parts."""
        payload: dict[str, object] = {"parts": ["invalid"]}

        assert _has_attachments(payload) is False

    def test_handles_non_list_headers(self) -> None:
        """Test handling non-list headers in part."""
        payload: dict[str, object] = {"parts": [{"mimeType": "text/plain", "headers": "invalid"}]}

        assert _has_attachments(payload) is False

    def test_handles_non_dict_header(self) -> None:
        """Test handling non-dict entries in part headers."""
        payload: dict[str, object] = {"parts": [{"mimeType": "text/plain", "headers": ["invalid"]}]}

        assert _has_attachments(payload) is False

    def test_detects_nested_attachments(self) -> None:
        """Test detecting attachments in nested parts."""
        payload = {
            "parts": [
                {
                    "mimeType": "multipart/mixed",
                    "parts": [{"mimeType": "application/pdf", "filename": "nested.pdf"}],
                }
            ]
        }

        assert _has_attachments(payload) is True

    def test_ignores_non_attachment_disposition(self) -> None:
        """Test that non-attachment Content-Disposition is ignored."""
        payload = {
            "parts": [
                {
                    "mimeType": "image/png",
                    "headers": [
                        {"name": "Content-Disposition", "value": "inline; filename=image.png"}
                    ],
                }
            ]
        }

        # inline disposition is not an attachment
        assert _has_attachments(payload) is False


def _encode_base64(text: str) -> str:
    """Helper to encode text to base64url format."""
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")
