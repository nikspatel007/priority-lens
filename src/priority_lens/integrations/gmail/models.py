"""Gmail API data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GmailMessageRef:
    """Reference to a Gmail message (from list operation).

    This lightweight object contains only the IDs needed to fetch
    the full message content.

    Attributes:
        id: Unique Gmail message ID.
        thread_id: ID of the thread this message belongs to.
    """

    id: str
    thread_id: str


@dataclass
class GmailMessage:
    """Full Gmail message with parsed data.

    This represents a complete Gmail message with all headers
    and body content parsed from the API response.

    Attributes:
        id: Unique Gmail message ID.
        thread_id: ID of the thread this message belongs to.
        history_id: Gmail history ID for incremental sync.
        message_id: RFC 822 Message-ID header value.
        subject: Email subject line.
        from_address: Sender email address.
        date_sent: When the message was sent.
        snippet: Short preview text from Gmail.
        size_bytes: Total message size in bytes.
        has_attachments: Whether the message has attachments.
        label_ids: List of Gmail label IDs applied to this message.
        to_addresses: List of recipient email addresses.
        cc_addresses: List of CC'd email addresses.
        in_reply_to: RFC 822 In-Reply-To header (if reply).
        references: List of RFC 822 References header values.
        body_plain: Plain text body (may be None).
        body_html: HTML body (may be None).
    """

    # Required fields (no defaults)
    id: str
    thread_id: str
    history_id: str
    message_id: str
    subject: str
    from_address: str
    date_sent: datetime
    snippet: str
    size_bytes: int
    has_attachments: bool

    # Fields with defaults
    label_ids: list[str] = field(default_factory=list)
    to_addresses: list[str] = field(default_factory=list)
    cc_addresses: list[str] = field(default_factory=list)
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)
    body_plain: str | None = None
    body_html: str | None = None

    def is_sent(self) -> bool:
        """Check if this message was sent by the user.

        Returns:
            True if the SENT label is present.
        """
        return "SENT" in self.label_ids

    def is_inbox(self) -> bool:
        """Check if this message is in the inbox.

        Returns:
            True if the INBOX label is present.
        """
        return "INBOX" in self.label_ids

    def is_unread(self) -> bool:
        """Check if this message is unread.

        Returns:
            True if the UNREAD label is present.
        """
        return "UNREAD" in self.label_ids

    def is_starred(self) -> bool:
        """Check if this message is starred.

        Returns:
            True if the STARRED label is present.
        """
        return "STARRED" in self.label_ids

    def is_draft(self) -> bool:
        """Check if this message is a draft.

        Returns:
            True if the DRAFT label is present.
        """
        return "DRAFT" in self.label_ids

    def is_spam(self) -> bool:
        """Check if this message is spam.

        Returns:
            True if the SPAM label is present.
        """
        return "SPAM" in self.label_ids

    def is_trash(self) -> bool:
        """Check if this message is in trash.

        Returns:
            True if the TRASH label is present.
        """
        return "TRASH" in self.label_ids
