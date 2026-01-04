#!/usr/bin/env python3
"""People score extraction for email prioritization.

Extracts features related to sender importance, relationship strength,
and organizational hierarchy from emails.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# Org hierarchy keywords by seniority level (higher = more senior)
# Longer phrases checked first to avoid partial matches (e.g., "vice president" before "president")
ORG_HIERARCHY_KEYWORDS = [
    (5, ['chief executive officer', 'ceo', 'chief operating officer', 'coo',
         'chief financial officer', 'cfo', 'chief technology officer', 'cto',
         'chief information officer', 'cio', 'chairman']),
    (4, ['executive vice president', 'evp', 'senior vice president', 'svp', 'partner']),
    (3, ['vice president', 'vp', 'director', 'head of', 'general counsel']),
    (5, ['president']),  # Check standalone "president" after "vice president"
    (2, ['manager', 'lead', 'senior', 'principal', 'supervisor']),
    (1, ['associate', 'analyst', 'specialist', 'coordinator', 'assistant']),
]


def extract_email_address(email_field: str) -> str:
    """Extract clean email address from header field.

    Handles formats like:
    - john.doe@enron.com
    - John Doe <john.doe@enron.com>
    - "Doe, John" <john.doe@enron.com>
    """
    if not email_field:
        return ''

    # Try to extract from angle brackets
    match = re.search(r'<([^>]+)>', email_field)
    if match:
        return match.group(1).lower().strip()

    # Otherwise assume it's just an email
    email_field = email_field.strip()
    if '@' in email_field:
        return email_field.lower()

    return ''


def extract_display_name(email_field: str) -> str:
    """Extract display name from email header field."""
    if not email_field:
        return ''

    # Check for "Name" <email> format
    match = re.match(r'^"?([^"<]+)"?\s*<', email_field)
    if match:
        return match.group(1).strip().strip('"')

    # Check for Name <email> without quotes
    if '<' in email_field:
        return email_field.split('<')[0].strip()

    return ''


def parse_email_list(email_field: str) -> list[str]:
    """Parse comma/semicolon separated email list."""
    if not email_field:
        return []

    # Split on comma or semicolon
    parts = re.split(r'[,;]', email_field)
    emails = []

    for part in parts:
        addr = extract_email_address(part.strip())
        if addr:
            emails.append(addr)

    return emails


def get_email_domain(email: str) -> str:
    """Extract domain from email address."""
    if '@' in email:
        return email.split('@')[1].lower()
    return ''


def detect_org_level(name: str) -> int:
    """Detect organizational level from display name or signature.

    Returns seniority level 0-5 (0 = unknown, 5 = C-level).
    """
    if not name:
        return 0

    name_lower = name.lower()

    # Check in order (longer phrases first to avoid partial matches)
    for level, keywords in ORG_HIERARCHY_KEYWORDS:
        for keyword in keywords:
            if keyword in name_lower:
                return level

    return 0


@dataclass
class SenderProfile:
    """Accumulated profile of a sender from interaction history."""
    email: str
    display_name: str = ''
    org_level: int = 0
    emails_received: int = 0
    emails_sent_to: int = 0
    replies_to_them: int = 0
    replies_from_them: int = 0
    first_interaction: Optional[str] = None
    last_interaction: Optional[str] = None
    domains: set = field(default_factory=set)


@dataclass
class PeopleFeatures:
    """People-related features for a single email."""

    # Sender features
    sender_email: str = ''
    sender_domain: str = ''
    sender_is_internal: bool = False
    sender_org_level: int = 0

    # Relationship features
    interaction_count: int = 0
    reply_rate: float = 0.0
    recency_days: float = float('inf')

    # Recipient position
    on_to: bool = False
    on_cc: bool = False
    on_bcc: bool = False
    recipient_count: int = 0

    # Thread features
    is_reply: bool = False
    in_active_thread: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'sender_email': self.sender_email,
            'sender_domain': self.sender_domain,
            'sender_is_internal': self.sender_is_internal,
            'sender_org_level': self.sender_org_level,
            'interaction_count': self.interaction_count,
            'reply_rate': self.reply_rate,
            'recency_days': self.recency_days if self.recency_days != float('inf') else -1,
            'on_to': self.on_to,
            'on_cc': self.on_cc,
            'on_bcc': self.on_bcc,
            'recipient_count': self.recipient_count,
            'is_reply': self.is_reply,
            'in_active_thread': self.in_active_thread,
        }

    def to_vector(self) -> list[float]:
        """Convert to feature vector for ML models."""
        return [
            1.0 if self.sender_is_internal else 0.0,
            float(self.sender_org_level) / 5.0,  # Normalized 0-1
            min(1.0, self.interaction_count / 100.0),  # Cap at 100
            self.reply_rate,
            1.0 / (1.0 + self.recency_days) if self.recency_days != float('inf') else 0.0,
            1.0 if self.on_to else 0.0,
            1.0 if self.on_cc else 0.0,
            1.0 if self.on_bcc else 0.0,
            min(1.0, self.recipient_count / 20.0),  # Normalized, cap at 20
            1.0 if self.is_reply else 0.0,
            1.0 if self.in_active_thread else 0.0,
        ]


class PeopleScorer:
    """Extract people-related features from emails.

    Maintains state about sender profiles built from email history.
    """

    def __init__(self, user_email: str, user_domain: str = 'enron.com'):
        """Initialize scorer.

        Args:
            user_email: The email address of the user whose inbox we're analyzing
            user_domain: The user's organization domain (internal emails)
        """
        self.user_email = user_email.lower()
        self.user_domain = user_domain.lower()
        self.sender_profiles: dict[str, SenderProfile] = defaultdict(
            lambda: SenderProfile(email='')
        )
        self.active_threads: set[str] = set()

    def build_profiles(self, emails: list[dict]) -> None:
        """Build sender profiles from email history.

        Call this first with historical emails before scoring new ones.

        Args:
            emails: List of email dictionaries with 'from', 'to', 'date', 'subject' fields
        """
        for email in emails:
            self._update_profile_from_email(email)

    def _update_profile_from_email(self, email: dict) -> None:
        """Update sender profiles from a single email."""
        sender = extract_email_address(email.get('from', ''))
        if not sender:
            return

        # Initialize or update profile
        if sender not in self.sender_profiles:
            self.sender_profiles[sender] = SenderProfile(email=sender)

        profile = self.sender_profiles[sender]

        # Update display name if available
        display_name = email.get('x_from', '') or extract_display_name(email.get('from', ''))
        if display_name and not profile.display_name:
            profile.display_name = display_name
            profile.org_level = max(profile.org_level, detect_org_level(display_name))

        # Track domain
        domain = get_email_domain(sender)
        if domain:
            profile.domains.add(domain)

        # Determine if this is received or sent by user
        to_list = parse_email_list(email.get('to', ''))
        cc_list = parse_email_list(email.get('cc', ''))

        if sender == self.user_email:
            # User sent this email
            for recipient in to_list + cc_list:
                if recipient != self.user_email:
                    if recipient not in self.sender_profiles:
                        self.sender_profiles[recipient] = SenderProfile(email=recipient)
                    self.sender_profiles[recipient].emails_sent_to += 1
        else:
            # User received this email
            profile.emails_received += 1

            # Check if it's a reply
            subject = email.get('subject', '').lower()
            if subject.startswith('re:'):
                profile.replies_from_them += 1

        # Track date
        date = email.get('date', '')
        if date:
            if not profile.first_interaction or date < profile.first_interaction:
                profile.first_interaction = date
            if not profile.last_interaction or date > profile.last_interaction:
                profile.last_interaction = date

        # Track active threads
        in_reply_to = email.get('in_reply_to', '')
        if in_reply_to:
            self.active_threads.add(in_reply_to)

        message_id = email.get('message_id', '')
        if message_id:
            self.active_threads.add(message_id)

    def extract_features(self, email: dict) -> PeopleFeatures:
        """Extract people features for a single email.

        Args:
            email: Email dictionary with standard fields

        Returns:
            PeopleFeatures dataclass with extracted features
        """
        features = PeopleFeatures()

        # Sender info
        sender = extract_email_address(email.get('from', ''))
        features.sender_email = sender
        features.sender_domain = get_email_domain(sender)
        features.sender_is_internal = features.sender_domain == self.user_domain

        # Org level from display name
        display_name = email.get('x_from', '') or extract_display_name(email.get('from', ''))
        features.sender_org_level = detect_org_level(display_name)

        # Get sender profile if available
        if sender in self.sender_profiles:
            profile = self.sender_profiles[sender]

            # Override org level from profile if higher
            features.sender_org_level = max(features.sender_org_level, profile.org_level)

            # Interaction count
            features.interaction_count = (
                profile.emails_received + profile.emails_sent_to
            )

            # Reply rate
            if profile.emails_received > 0:
                features.reply_rate = profile.replies_to_them / profile.emails_received

        # Recipient position
        to_list = parse_email_list(email.get('to', ''))
        cc_list = parse_email_list(email.get('cc', ''))
        bcc_list = parse_email_list(email.get('bcc', ''))

        features.on_to = self.user_email in to_list
        features.on_cc = self.user_email in cc_list
        features.on_bcc = self.user_email in bcc_list
        features.recipient_count = len(to_list) + len(cc_list) + len(bcc_list)

        # Thread features
        subject = email.get('subject', '').lower()
        features.is_reply = subject.startswith('re:') or subject.startswith('fw:')

        in_reply_to = email.get('in_reply_to', '')
        features.in_active_thread = in_reply_to in self.active_threads

        return features


def score_email_people(
    email: dict,
    scorer: PeopleScorer
) -> dict:
    """Extract people score features for an email.

    Convenience function that combines extraction and scoring.

    Args:
        email: Email dictionary
        scorer: Initialized PeopleScorer with built profiles

    Returns:
        Dictionary of people features
    """
    features = scorer.extract_features(email)
    return features.to_dict()


if __name__ == '__main__':
    # Demo usage
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Extract people features from emails')
    parser.add_argument('input', help='Input emails JSON file')
    parser.add_argument('--user', required=True, help='User email address')
    parser.add_argument('--limit', type=int, help='Limit emails to process')
    args = parser.parse_args()

    with open(args.input) as f:
        emails = json.load(f)

    if args.limit:
        emails = emails[:args.limit]

    # Initialize scorer and build profiles
    scorer = PeopleScorer(user_email=args.user)
    scorer.build_profiles(emails)

    print(f"Built profiles for {len(scorer.sender_profiles)} senders")
    print()

    # Extract features for each email
    for email in emails[:5]:  # Show first 5
        features = scorer.extract_features(email)
        print(f"From: {features.sender_email}")
        print(f"  Internal: {features.sender_is_internal}")
        print(f"  Org level: {features.sender_org_level}")
        print(f"  Interactions: {features.interaction_count}")
        print(f"  On TO: {features.on_to}, CC: {features.on_cc}")
        print()
