"""Email content sanitization module.

Provides comprehensive email cleaning for ML/embedding preparation:
- HTML to plain text conversion
- Template syntax removal (Liquid, Jinja, Handlebars)
- Quoted reply chain removal
- Signature detection and removal
- Tracking pixel and invisible content removal
- Character normalization
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment


@dataclass
class SanitizationResult:
    """Result of email sanitization."""

    text: str
    original_length: int
    sanitized_length: int
    html_stripped: bool
    quotes_removed: bool
    signature_removed: bool

    @property
    def reduction_ratio(self) -> float:
        """Calculate content reduction ratio."""
        if self.original_length == 0:
            return 0.0
        return 1.0 - (self.sanitized_length / self.original_length)


# Patterns for template syntax removal
TEMPLATE_PATTERNS = [
    # Liquid/Jinja block tags: {% ... %}
    re.compile(r"\{%.*?%\}", re.DOTALL),
    # Liquid/Jinja variables: {{ ... }}
    re.compile(r"\{\{.*?\}\}", re.DOTALL),
    # Jinja comments: {# ... #}
    re.compile(r"\{#.*?#\}", re.DOTALL),
    # Handlebars: {{# ... }} and {{/ ... }}
    re.compile(r"\{\{[#/].*?\}\}", re.DOTALL),
    # ERB tags: <% ... %>
    re.compile(r"<%.*?%>", re.DOTALL),
    # Mustache sections: {{#section}}...{{/section}}
    re.compile(r"\{\{[#^/].*?\}\}", re.DOTALL),
]

# Patterns for quote detection
QUOTE_PATTERNS = [
    # Standard quoted lines
    re.compile(r"^>+\s*", re.MULTILINE),
    # "On ... wrote:" pattern
    re.compile(r"^On .+ wrote:$", re.MULTILINE),
    # Outlook style "From: ... Sent: ..."
    re.compile(r"^(From|Sent|To|Subject|Date|Cc|Bcc):\s*.+$", re.MULTILINE),
    # Gmail forwarded message header
    re.compile(r"^-+\s*Forwarded message\s*-+$", re.MULTILINE),
    # Original message header
    re.compile(r"^-+\s*Original Message\s*-+$", re.MULTILINE | re.IGNORECASE),
]

# Signature markers
SIGNATURE_MARKERS = [
    "--",
    "---",
    "-- ",
    "—",  # em dash
    "Sent from my iPhone",
    "Sent from my iPad",
    "Sent from my Android",
    "Sent from Mobile",
    "Get Outlook for iOS",
    "Get Outlook for Android",
    "Sent via ",
    "Sent from Yahoo Mail",
    "Sent from Gmail",
    "Best regards,",
    "Best,",
    "Thanks,",
    "Regards,",
    "Cheers,",
    "Sincerely,",
]

# Elements to completely remove from HTML
REMOVE_ELEMENTS = [
    "script",
    "style",
    "head",
    "meta",
    "link",
    "noscript",
    "iframe",
    "object",
    "embed",
    "applet",
    # Tracking pixels
    "img[width='1']",
    "img[height='1']",
    "img[style*='display:none']",
    "img[style*='display: none']",
]


def strip_html(html: str) -> str:
    """Convert HTML email to plain text.

    Removes scripts, styles, tracking pixels, and invisible elements.

    Args:
        html: Raw HTML content.

    Returns:
        Plain text content.
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove unwanted elements
        for selector in REMOVE_ELEMENTS:
            if "[" in selector:
                # Handle attribute selectors
                for element in soup.select(selector):
                    element.decompose()
            else:
                for element in soup.find_all(selector):
                    element.decompose()

        # Remove hidden elements
        for element in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
            element.decompose()
        for element in soup.find_all(style=re.compile(r"visibility:\s*hidden", re.I)):
            element.decompose()

        # Note: 1x1 tracking pixels are already handled by CSS selectors
        # img[width='1'] and img[height='1'] in REMOVE_ELEMENTS

        # Extract text with proper spacing
        text = soup.get_text(separator=" ")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    except Exception:
        # If parsing fails, return the original with basic tag stripping
        return re.sub(r"<[^>]+>", " ", html)


def strip_template_syntax(text: str) -> str:
    """Remove Liquid, Jinja, Handlebars, ERB template syntax.

    Args:
        text: Text with potential template syntax.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    for pattern in TEMPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Clean up resulting whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_quoted_content(text: str) -> str:
    """Remove quoted reply chains from email text.

    Handles both line-based quotes and inline quotes.

    Args:
        text: Email text with potential quotes.

    Returns:
        Text without quoted replies.
    """
    if not text:
        return ""

    # First, handle inline quotes (from HTML stripping that collapsed newlines)
    # Remove "On ... wrote: > quoted text" patterns
    text = re.sub(r"On [^>]+wrote:\s*>.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Note: Removed problematic regex r"\s*>\s+[^>]+(?:\s*>\s*[^>]+)*$" that caused
    # catastrophic backtracking on text with many '>' characters on a single line.
    # The line-based processing below handles quoted content adequately.

    # Now handle line-based quotes
    lines = text.split("\n")
    clean_lines = []
    in_quote_block = False

    for line in lines:
        stripped = line.strip()

        # Check for quote markers
        if stripped.startswith(">"):
            in_quote_block = True
            continue

        # Check for "On ... wrote:" pattern
        if re.match(r"^On .+ wrote:$", stripped, re.IGNORECASE):
            in_quote_block = True
            continue

        # Check for forwarded/original message headers
        if re.match(r"^-+\s*(Forwarded message|Original Message)\s*-+$", stripped, re.I):
            in_quote_block = True
            continue

        # Check for Outlook-style headers (From:, Sent:, etc.)
        if re.match(r"^(From|Sent|To|Subject|Date|Cc|Bcc):\s*.+$", stripped):
            continue  # Skip these lines but don't enter quote block

        # If we're in a quote block and see a non-quoted line, exit the block
        if in_quote_block and stripped and not stripped.startswith(">"):
            # Check if this looks like more quoted content
            if not any(stripped.startswith(marker) for marker in [">", ">>", ">>>"]):
                in_quote_block = False

        if not in_quote_block:
            clean_lines.append(line)

    result = "\n".join(clean_lines).strip()

    # Final cleanup: remove any remaining "> text" at end of lines
    result = re.sub(r"\s*>\s+\S.*$", "", result, flags=re.MULTILINE)

    return result.strip()


def remove_signature(text: str) -> tuple[str, bool]:
    """Remove email signature from text.

    Args:
        text: Email text.

    Returns:
        Tuple of (cleaned text, whether signature was removed).
    """
    if not text:
        return "", False

    lines = text.split("\n")
    signature_start = -1

    # Find signature start marker
    for i, line in enumerate(lines):
        stripped = line.strip()
        for marker in SIGNATURE_MARKERS:
            if stripped == marker or stripped.startswith(marker):
                signature_start = i
                break
        if signature_start >= 0:
            break

    if signature_start >= 0:
        # Keep only lines before signature
        return "\n".join(lines[:signature_start]).strip(), True

    return text, False


def normalize_characters(text: str) -> str:
    """Normalize Unicode characters for consistent processing.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text.
    """
    if not text:
        return ""

    # Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Replace smart double quotes with standard quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # " and "

    # Replace smart single quotes with standard quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # ' and '

    # Replace em/en dashes with hyphens
    text = text.replace("\u2014", "-").replace("\u2013", "-")  # — and –

    # Replace ellipsis with three dots
    text = text.replace("\u2026", "...")  # …

    # Remove zero-width characters
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)

    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to single spaces.

    Args:
        text: Text to clean.

    Returns:
        Text with collapsed whitespace.
    """
    if not text:
        return ""

    # Replace multiple newlines with double newline (preserve paragraphs)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # Replace multiple spaces/tabs with single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def sanitize_email(
    content: str,
    strip_html_content: bool = True,
    remove_templates: bool = True,
    remove_quotes: bool = True,
    remove_sig: bool = True,
    normalize_chars: bool = True,
) -> SanitizationResult:
    """Comprehensive email sanitization for ML/embedding preparation.

    Args:
        content: Raw email content (HTML or plain text).
        strip_html_content: Whether to strip HTML tags.
        remove_templates: Whether to remove template syntax.
        remove_quotes: Whether to remove quoted replies.
        remove_sig: Whether to remove signatures.
        normalize_chars: Whether to normalize Unicode characters.

    Returns:
        SanitizationResult with cleaned text and statistics.
    """
    if not content:
        return SanitizationResult(
            text="",
            original_length=0,
            sanitized_length=0,
            html_stripped=False,
            quotes_removed=False,
            signature_removed=False,
        )

    original_length = len(content)
    html_stripped = False
    quotes_removed = False
    signature_removed = False

    text = content

    # Step 1: Strip HTML if present
    text_lower = text.lower()
    has_html = any(
        tag in text_lower
        for tag in ["<html", "<body", "<p", "<div", "<script", "<style", "<table", "<span"]
    )
    if strip_html_content and has_html:
        text = strip_html(text)
        html_stripped = True

    # Step 2: Remove template syntax
    if remove_templates:
        text = strip_template_syntax(text)

    # Step 3: Remove quoted content
    if remove_quotes:
        before_len = len(text)
        text = remove_quoted_content(text)
        quotes_removed = len(text) < before_len

    # Step 4: Remove signature
    if remove_sig:
        text, signature_removed = remove_signature(text)

    # Step 5: Normalize characters
    if normalize_chars:
        text = normalize_characters(text)

    # Step 6: Collapse whitespace
    text = collapse_whitespace(text)

    return SanitizationResult(
        text=text,
        original_length=original_length,
        sanitized_length=len(text),
        html_stripped=html_stripped,
        quotes_removed=quotes_removed,
        signature_removed=signature_removed,
    )
