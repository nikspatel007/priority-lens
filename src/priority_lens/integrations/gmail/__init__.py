"""Gmail API integration."""

from priority_lens.integrations.gmail.client import GmailApiError, GmailClient
from priority_lens.integrations.gmail.models import GmailMessage, GmailMessageRef
from priority_lens.integrations.gmail.parser import (
    GmailParseError,
    gmail_to_email_data,
    parse_email_address,
    parse_raw_message,
)
from priority_lens.integrations.gmail.rate_limiter import RateLimiter

__all__ = [
    "GmailApiError",
    "GmailClient",
    "GmailMessage",
    "GmailMessageRef",
    "GmailParseError",
    "RateLimiter",
    "gmail_to_email_data",
    "parse_email_address",
    "parse_raw_message",
]
