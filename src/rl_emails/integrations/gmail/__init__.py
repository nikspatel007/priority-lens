"""Gmail API integration."""

from rl_emails.integrations.gmail.client import GmailApiError, GmailClient
from rl_emails.integrations.gmail.models import GmailMessage, GmailMessageRef
from rl_emails.integrations.gmail.parser import (
    GmailParseError,
    gmail_to_email_data,
    parse_email_address,
    parse_raw_message,
)
from rl_emails.integrations.gmail.rate_limiter import RateLimiter

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
