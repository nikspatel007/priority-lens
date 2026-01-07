"""Gmail message parser for converting to internal EmailData format."""

from __future__ import annotations

import base64
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rl_emails.core.types import EmailData
from rl_emails.integrations.gmail.models import GmailMessage

if TYPE_CHECKING:
    pass


class GmailParseError(Exception):
    """Exception raised when parsing Gmail messages fails.

    Attributes:
        message_id: Gmail message ID that failed to parse.
        field: The field that caused the parsing error.
        reason: Human-readable error description.
    """

    def __init__(
        self,
        reason: str,
        message_id: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.message_id = message_id
        self.field = field
        self.reason = reason


def parse_email_address(address: str) -> tuple[str, str | None]:
    """Parse an email address into email and name components.

    Handles formats like:
    - "Name <email@example.com>"
    - "<email@example.com>"
    - "email@example.com"

    Args:
        address: Raw email address string.

    Returns:
        Tuple of (email, name) where name may be None.
    """
    if not address:
        return "", None

    # Match "Name <email>" format
    match = re.match(r'^"?([^"<]+)"?\s*<([^>]+)>$', address.strip())
    if match:
        name = match.group(1).strip()
        email = match.group(2).strip().lower()
        return email, name if name else None

    # Match "<email>" format
    match = re.match(r"^<([^>]+)>$", address.strip())
    if match:
        return match.group(1).strip().lower(), None

    # Bare email address
    if "@" in address:
        return address.strip().lower(), None

    return address.strip(), None


def parse_raw_message(raw_data: dict[str, object]) -> GmailMessage:
    """Parse raw Gmail API response into GmailMessage.

    This parses the JSON response from Gmail's messages.get API
    into our GmailMessage dataclass.

    Args:
        raw_data: Raw JSON response from Gmail API.

    Returns:
        Parsed GmailMessage object.

    Raises:
        GmailParseError: If required fields are missing or invalid.
    """
    msg_id = str(raw_data.get("id", ""))
    if not msg_id:
        raise GmailParseError("Missing message ID", field="id")

    thread_id = str(raw_data.get("threadId", ""))
    history_id = str(raw_data.get("historyId", ""))
    label_ids_raw = raw_data.get("labelIds", [])
    label_ids = list(label_ids_raw) if isinstance(label_ids_raw, list) else []
    snippet = str(raw_data.get("snippet", ""))
    size_raw = raw_data.get("sizeEstimate", 0)
    size_bytes = int(size_raw) if isinstance(size_raw, int) else 0

    # Parse payload
    payload = raw_data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    headers = _parse_headers(payload)

    # Extract header values
    message_id = headers.get("message-id", f"<{msg_id}@gmail.com>")
    subject = headers.get("subject", "")
    from_address = headers.get("from", "")
    to_raw = headers.get("to", "")
    cc_raw = headers.get("cc", "")
    date_str = headers.get("date", "")
    in_reply_to = headers.get("in-reply-to")
    references_raw = headers.get("references", "")

    # Parse addresses
    to_addresses = _split_addresses(to_raw)
    cc_addresses = _split_addresses(cc_raw)
    references = _parse_references(references_raw)

    # Parse date

    date_sent = _parse_date(date_str)

    # Parse body
    body_plain, body_html = _parse_body(payload)

    # Check for attachments
    has_attachments = _has_attachments(payload)

    return GmailMessage(
        id=msg_id,
        thread_id=thread_id,
        history_id=history_id,
        message_id=message_id,
        subject=subject,
        from_address=from_address,
        date_sent=date_sent,
        snippet=snippet,
        size_bytes=size_bytes,
        has_attachments=has_attachments,
        label_ids=label_ids,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        in_reply_to=in_reply_to,
        references=references,
        body_plain=body_plain,
        body_html=body_html,
    )


def gmail_to_email_data(gmail_msg: GmailMessage) -> EmailData:
    """Convert Gmail message to internal EmailData format.

    This bridges the Gmail API format to the format expected by
    pipeline stages 3-11.

    Args:
        gmail_msg: Parsed Gmail message.

    Returns:
        EmailData dict for pipeline processing.
    """
    from_email, from_name = parse_email_address(gmail_msg.from_address)

    # Build headers dict for compatibility
    headers: dict[str, str] = {
        "message-id": gmail_msg.message_id,
        "from": gmail_msg.from_address,
        "subject": gmail_msg.subject,
        "date": gmail_msg.date_sent.isoformat(),
    }

    if gmail_msg.to_addresses:
        headers["to"] = ", ".join(gmail_msg.to_addresses)
    if gmail_msg.cc_addresses:
        headers["cc"] = ", ".join(gmail_msg.cc_addresses)
    if gmail_msg.in_reply_to:
        headers["in-reply-to"] = gmail_msg.in_reply_to
    if gmail_msg.references:
        headers["references"] = " ".join(gmail_msg.references)

    return EmailData(
        message_id=gmail_msg.message_id,
        from_email=from_email,
        from_name=from_name,
        to_emails=gmail_msg.to_addresses,
        cc_emails=gmail_msg.cc_addresses,
        bcc_emails=[],
        subject=gmail_msg.subject,
        date_str=gmail_msg.date_sent.isoformat(),
        body_text=gmail_msg.body_plain or "",
        body_html=gmail_msg.body_html,
        headers=headers,
        labels=gmail_msg.label_ids,
        in_reply_to=gmail_msg.in_reply_to,
        references=gmail_msg.references,
    )


def _parse_headers(payload: dict[str, object]) -> dict[str, str]:
    """Parse headers from payload into lowercase-keyed dict."""
    headers: dict[str, str] = {}
    raw_headers = payload.get("headers", [])

    if not isinstance(raw_headers, list):
        return headers

    for header in raw_headers:
        if isinstance(header, dict):
            name = str(header.get("name", "")).lower()
            value = str(header.get("value", ""))
            if name:
                headers[name] = value

    return headers


def _split_addresses(raw: str) -> list[str]:
    """Split comma-separated email addresses."""
    if not raw:
        return []

    addresses = []
    # Split on comma, but handle quoted names with commas
    parts = re.split(r",\s*(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", raw)
    for part in parts:
        part = part.strip()
        if part:
            email, _ = parse_email_address(part)
            if email:
                addresses.append(email)

    return addresses


def _parse_references(raw: str) -> list[str]:
    """Parse References header into list of message IDs."""
    if not raw:
        return []

    # References are space-separated message IDs
    refs = []
    for ref in raw.split():
        ref = ref.strip()
        if ref.startswith("<") and ref.endswith(">"):
            refs.append(ref)
        elif "@" in ref:
            refs.append(f"<{ref}>")

    return refs


def _parse_date(date_str: str) -> datetime:
    """Parse email date string to datetime."""
    from email.utils import parsedate_to_datetime

    if not date_str:
        return datetime.now(UTC)

    try:
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        pass

    # Fallback: try ISO format
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass

    # Last resort: return current time
    return datetime.now(UTC)


def _parse_body(payload: dict[str, object]) -> tuple[str | None, str | None]:
    """Parse body content from payload.

    Gmail API returns body in base64url encoded format.
    Handles multipart messages by recursively searching parts.

    Returns:
        Tuple of (plain_text, html) content.
    """
    body_plain: str | None = None
    body_html: str | None = None

    mime_type = str(payload.get("mimeType", ""))

    # Direct body content
    body = payload.get("body", {})
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, str):
            decoded = _decode_base64(data)
            if "text/plain" in mime_type:
                body_plain = decoded
            elif "text/html" in mime_type:
                body_html = decoded

    # Check parts for multipart messages
    parts = payload.get("parts", [])
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict):
                part_plain, part_html = _parse_body(part)
                if part_plain and not body_plain:
                    body_plain = part_plain
                if part_html and not body_html:
                    body_html = part_html

    return body_plain, body_html


def _decode_base64(data: str) -> str:
    """Decode base64url encoded string."""
    try:
        # Gmail uses URL-safe base64 without padding
        padded = data + "=" * (4 - len(data) % 4)
        decoded_bytes = base64.urlsafe_b64decode(padded)
        return decoded_bytes.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover
        return ""  # pragma: no cover


def _has_attachments(payload: dict[str, object]) -> bool:
    """Check if payload contains attachments."""
    parts = payload.get("parts", [])
    if not isinstance(parts, list):
        return False

    for part in parts:
        if not isinstance(part, dict):
            continue

        # Check for attachment disposition
        filename = part.get("filename", "")
        if filename:
            return True

        # Check Content-Disposition header
        headers = part.get("headers", [])
        if isinstance(headers, list):
            for header in headers:
                if isinstance(header, dict):
                    name = str(header.get("name", "")).lower()
                    value = str(header.get("value", "")).lower()
                    if name == "content-disposition" and "attachment" in value:
                        return True

        # Recursively check nested parts
        if _has_attachments(part):
            return True

    return False
