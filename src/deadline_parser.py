#!/usr/bin/env python3
"""Enhanced deadline parsing with traditional + LLM extraction.

This module provides comprehensive deadline extraction from email text:
- Traditional regex-based pattern matching for explicit dates
- LLM-based semantic extraction for implicit/contextual deadlines
- Resolution of relative dates (EOD, this week, ASAP) to datetime
- Handling of business calendar awareness

Usage:
    from deadline_parser import DeadlineParser

    parser = DeadlineParser()
    deadlines = parser.extract(email_text)
    for d in deadlines:
        print(f"{d.text} -> {d.resolved_date} (confidence: {d.confidence})")
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Optional, Callable
import calendar


class DeadlineType(Enum):
    """Categories of deadline expressions."""
    EXPLICIT_DATE = "explicit_date"      # Jan 15, 12/25/2026
    EXPLICIT_TIME = "explicit_time"      # 3pm, 15:00
    DAY_NAME = "day_name"                # Monday, Friday
    RELATIVE_DAY = "relative_day"        # today, tomorrow, tonight
    PERIOD_END = "period_end"            # end of day, end of week
    ACRONYM = "acronym"                  # EOD, EOB, COB, EOW, EOM, EOQ, EOY
    URGENCY = "urgency"                  # ASAP, urgent, immediately
    RELATIVE_PERIOD = "relative_period"  # next week, in 2 days
    CONTEXTUAL = "contextual"            # before the meeting, by the deadline


class UrgencyLevel(Enum):
    """Urgency levels for deadlines."""
    CRITICAL = 1.0      # ASAP, urgent, immediately
    HIGH = 0.9          # today, tonight, EOD
    MEDIUM_HIGH = 0.7   # tomorrow
    MEDIUM = 0.5        # this week, by Friday
    LOW = 0.3           # end of month, next week
    MINIMAL = 0.1       # no specific deadline


@dataclass
class Deadline:
    """A parsed deadline with resolved datetime."""
    text: str                           # Original text that triggered detection
    deadline_type: DeadlineType         # Category of deadline
    resolved_date: Optional[datetime]   # Resolved to actual datetime
    urgency: float                      # 0-1 urgency score
    confidence: float                   # 0-1 extraction confidence
    start_pos: int                      # Start position in text
    end_pos: int                        # End position in text
    source: str = "traditional"         # "traditional" or "llm"

    def days_until(self, reference: Optional[datetime] = None) -> Optional[float]:
        """Calculate days until this deadline."""
        if not self.resolved_date:
            return None
        ref = reference or datetime.now()
        delta = self.resolved_date - ref
        return delta.total_seconds() / 86400


@dataclass
class DeadlineParserConfig:
    """Configuration for deadline parsing behavior."""
    # Business hours
    business_start: time = field(default_factory=lambda: time(9, 0))
    business_end: time = field(default_factory=lambda: time(17, 0))

    # Default times for unspecified deadlines
    eod_time: time = field(default_factory=lambda: time(17, 0))
    morning_time: time = field(default_factory=lambda: time(9, 0))

    # LLM settings
    use_llm: bool = True
    llm_model: str = "all-MiniLM-L6-v2"
    llm_threshold: float = 0.6

    # Calendar awareness
    skip_weekends: bool = True
    holidays: list[datetime] = field(default_factory=list)


# Day name to weekday number (Monday=0, Sunday=6)
DAY_NAMES = {
    'monday': 0, 'mon': 0,
    'tuesday': 1, 'tue': 1, 'tues': 1,
    'wednesday': 2, 'wed': 2,
    'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
    'friday': 4, 'fri': 4,
    'saturday': 5, 'sat': 5,
    'sunday': 6, 'sun': 6,
}

# Month name to number
MONTH_NAMES = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}

# Urgency keywords and their levels
URGENCY_KEYWORDS = {
    'asap': UrgencyLevel.CRITICAL,
    'urgent': UrgencyLevel.CRITICAL,
    'urgently': UrgencyLevel.CRITICAL,
    'immediately': UrgencyLevel.CRITICAL,
    'right away': UrgencyLevel.CRITICAL,
    'critical': UrgencyLevel.CRITICAL,
    'time-sensitive': UrgencyLevel.CRITICAL,
    'time sensitive': UrgencyLevel.CRITICAL,
    'as soon as possible': UrgencyLevel.CRITICAL,
    'high priority': UrgencyLevel.HIGH,
    'important': UrgencyLevel.MEDIUM_HIGH,
}

# Implicit deadline patterns and their meanings
IMPLICIT_PATTERNS = [
    # Period endings
    (r'\b(?:by\s+)?(?:the\s+)?end\s+of\s+(?:the\s+)?(day|today)\b', 'eod'),
    (r'\b(?:by\s+)?(?:the\s+)?end\s+of\s+(?:the\s+)?(week|this\s+week)\b', 'eow'),
    (r'\b(?:by\s+)?(?:the\s+)?end\s+of\s+(?:the\s+)?(month|this\s+month)\b', 'eom'),
    (r'\b(?:by\s+)?(?:the\s+)?end\s+of\s+(?:the\s+)?(quarter|this\s+quarter|q[1-4])\b', 'eoq'),
    (r'\b(?:by\s+)?(?:the\s+)?end\s+of\s+(?:the\s+)?(year|this\s+year)\b', 'eoy'),

    # Acronyms
    (r'\bEOD\b', 'eod'),
    (r'\bEOB\b', 'eob'),
    (r'\bCOB\b', 'cob'),
    (r'\bEOW\b', 'eow'),
    (r'\bEOM\b', 'eom'),
    (r'\bEOQ\b', 'eoq'),
    (r'\bEOY\b', 'eoy'),

    # Relative day references
    (r'\b(?:by\s+)?today\b', 'today'),
    (r'\b(?:by\s+)?tonight\b', 'tonight'),
    (r'\b(?:by\s+)?tomorrow\b', 'tomorrow'),
    (r'\b(?:by\s+)?tomorrow\s+morning\b', 'tomorrow_morning'),
    (r'\b(?:by\s+)?tomorrow\s+(?:afternoon|evening)\b', 'tomorrow_afternoon'),

    # Relative week references
    (r'\b(?:by\s+)?this\s+week\b', 'this_week'),
    (r'\b(?:by\s+)?next\s+week\b', 'next_week'),
    (r'\b(?:by\s+)?(?:the\s+)?beginning\s+of\s+(?:next\s+)?week\b', 'beginning_next_week'),
    (r'\b(?:by\s+)?(?:the\s+)?start\s+of\s+(?:next\s+)?week\b', 'beginning_next_week'),

    # Relative time expressions
    (r'\bin\s+(\d+)\s+hour(?:s)?\b', 'in_hours'),
    (r'\bin\s+(\d+)\s+day(?:s)?\b', 'in_days'),
    (r'\bin\s+(\d+)\s+week(?:s)?\b', 'in_weeks'),
    (r'\bwithin\s+(\d+)\s+hour(?:s)?\b', 'in_hours'),
    (r'\bwithin\s+(\d+)\s+day(?:s)?\b', 'in_days'),
    (r'\bwithin\s+(\d+)\s+week(?:s)?\b', 'in_weeks'),
    (r'\bwithin\s+(?:the\s+)?(?:next\s+)?(\d+)\s+hour(?:s)?\b', 'in_hours'),
    (r'\bwithin\s+(?:the\s+)?(?:next\s+)?(\d+)\s+day(?:s)?\b', 'in_days'),

    # Sprint/cycle references (common in software dev)
    (r'\b(?:by\s+)?(?:the\s+)?end\s+of\s+(?:this\s+)?sprint\b', 'end_of_sprint'),
    (r'\b(?:by\s+)?next\s+sprint\b', 'next_sprint'),
]

# Explicit date patterns
EXPLICIT_DATE_PATTERNS = [
    # MM/DD or M/D
    (r'\b(\d{1,2})/(\d{1,2})\b', 'slash_date'),
    # MM/DD/YYYY or MM/DD/YY
    (r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', 'slash_date_year'),
    # MM-DD or M-D
    (r'\b(\d{1,2})-(\d{1,2})\b(?!/)', 'dash_date'),
    # MM-DD-YYYY
    (r'\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b', 'dash_date_year'),
    # YYYY-MM-DD (ISO format)
    (r'\b(\d{4})-(\d{2})-(\d{2})\b', 'iso_date'),
    # Month DD or Month DDth
    (r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s+(\d{1,2})(?:st|nd|rd|th)?\b', 'month_day'),
    # DD Month
    (r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b', 'day_month'),
]

# Day name patterns
DAY_PATTERNS = [
    (r'\b(?:by|before|on|this|next)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'day_reference'),
    (r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(?:morning|afternoon|evening)\b', 'day_time'),
]

# Time patterns
TIME_PATTERNS = [
    (r'\b(?:by|at|before)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)\b', 'time_12h'),
    (r'\b(?:by|at|before)\s+(\d{1,2}):(\d{2})\b', 'time_24h'),
    (r'\b(\d{1,2})\s*(am|pm|AM|PM)\b', 'time_simple'),
]


class DeadlineParser:
    """Enhanced deadline parser with traditional + LLM extraction."""

    def __init__(self, config: Optional[DeadlineParserConfig] = None):
        """Initialize parser with configuration.

        Args:
            config: Parser configuration, uses defaults if not provided
        """
        self.config = config or DeadlineParserConfig()
        self._llm_model = None
        self._deadline_embeddings = None

    def _init_llm(self) -> bool:
        """Initialize LLM model for semantic extraction.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.config.use_llm:
            return False

        if self._llm_model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            self._llm_model = SentenceTransformer(self.config.llm_model)

            # Pre-compute embeddings for deadline-related phrases
            deadline_phrases = [
                "needs to be done by",
                "deadline is",
                "due date",
                "must be completed by",
                "required by",
                "expecting this by",
                "should have this done by",
                "time-sensitive",
                "urgent deadline",
                "submit before",
                "finish before",
                "complete by",
            ]
            self._deadline_embeddings = self._llm_model.encode(
                deadline_phrases, convert_to_numpy=True
            )
            return True
        except ImportError:
            self.config.use_llm = False
            return False

    def extract(
        self,
        text: str,
        reference_date: Optional[datetime] = None
    ) -> list[Deadline]:
        """Extract all deadlines from text.

        Args:
            text: Text to extract deadlines from
            reference_date: Reference date for relative calculations (defaults to now)

        Returns:
            List of extracted Deadline objects, sorted by urgency (highest first)
        """
        ref = reference_date or datetime.now()
        deadlines = []

        # Traditional extraction
        deadlines.extend(self._extract_urgency_keywords(text))
        deadlines.extend(self._extract_implicit_deadlines(text, ref))
        deadlines.extend(self._extract_day_references(text, ref))
        deadlines.extend(self._extract_explicit_dates(text, ref))
        deadlines.extend(self._extract_times(text, ref))

        # LLM extraction (if enabled)
        if self.config.use_llm and self._init_llm():
            llm_deadlines = self._extract_llm_deadlines(text, ref)
            deadlines.extend(llm_deadlines)

        # Deduplicate and merge overlapping deadlines
        deadlines = self._deduplicate_deadlines(deadlines)

        # Sort by urgency (highest first)
        deadlines.sort(key=lambda d: (-d.urgency, d.start_pos))

        return deadlines

    def _extract_urgency_keywords(self, text: str) -> list[Deadline]:
        """Extract urgency keyword deadlines (ASAP, urgent, etc.)."""
        deadlines = []
        text_lower = text.lower()

        for keyword, level in URGENCY_KEYWORDS.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, text_lower):
                deadlines.append(Deadline(
                    text=match.group(0),
                    deadline_type=DeadlineType.URGENCY,
                    resolved_date=None,  # Urgency keywords don't resolve to specific dates
                    urgency=level.value,
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        return deadlines

    def _extract_implicit_deadlines(
        self,
        text: str,
        ref: datetime
    ) -> list[Deadline]:
        """Extract implicit deadline patterns (EOD, this week, etc.)."""
        deadlines = []
        text_lower = text.lower()

        for pattern, deadline_key in IMPLICIT_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                resolved, urgency = self._resolve_implicit_deadline(
                    deadline_key, match, ref
                )

                deadlines.append(Deadline(
                    text=match.group(0),
                    deadline_type=self._get_deadline_type(deadline_key),
                    resolved_date=resolved,
                    urgency=urgency,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        return deadlines

    def _resolve_implicit_deadline(
        self,
        key: str,
        match: re.Match,
        ref: datetime
    ) -> tuple[Optional[datetime], float]:
        """Resolve implicit deadline to datetime and urgency.

        Returns:
            Tuple of (resolved_datetime, urgency_score)
        """
        eod = ref.replace(
            hour=self.config.eod_time.hour,
            minute=self.config.eod_time.minute,
            second=0, microsecond=0
        )
        morning = ref.replace(
            hour=self.config.morning_time.hour,
            minute=self.config.morning_time.minute,
            second=0, microsecond=0
        )

        if key in ('eod', 'eob', 'cob', 'today'):
            return eod, UrgencyLevel.HIGH.value

        elif key == 'tonight':
            tonight = ref.replace(hour=21, minute=0, second=0, microsecond=0)
            return tonight, UrgencyLevel.HIGH.value

        elif key == 'tomorrow':
            tomorrow_eod = (ref + timedelta(days=1)).replace(
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            return tomorrow_eod, UrgencyLevel.MEDIUM_HIGH.value

        elif key == 'tomorrow_morning':
            tomorrow_morning = (ref + timedelta(days=1)).replace(
                hour=self.config.morning_time.hour,
                minute=self.config.morning_time.minute,
                second=0, microsecond=0
            )
            return tomorrow_morning, UrgencyLevel.MEDIUM_HIGH.value

        elif key == 'tomorrow_afternoon':
            tomorrow_afternoon = (ref + timedelta(days=1)).replace(
                hour=14, minute=0, second=0, microsecond=0
            )
            return tomorrow_afternoon, UrgencyLevel.MEDIUM_HIGH.value

        elif key in ('eow', 'this_week'):
            # Find Friday at EOD
            days_until_friday = (4 - ref.weekday()) % 7
            if days_until_friday == 0 and ref.hour >= self.config.eod_time.hour:
                days_until_friday = 7
            friday = ref + timedelta(days=days_until_friday)
            friday_eod = friday.replace(
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            return friday_eod, UrgencyLevel.MEDIUM.value

        elif key == 'next_week':
            # Monday of next week at 9am
            days_until_monday = (7 - ref.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            monday = ref + timedelta(days=days_until_monday)
            monday_morning = monday.replace(
                hour=self.config.morning_time.hour,
                minute=self.config.morning_time.minute,
                second=0, microsecond=0
            )
            return monday_morning, UrgencyLevel.LOW.value

        elif key == 'beginning_next_week':
            days_until_monday = (7 - ref.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            monday = ref + timedelta(days=days_until_monday)
            monday_morning = monday.replace(
                hour=self.config.morning_time.hour,
                minute=self.config.morning_time.minute,
                second=0, microsecond=0
            )
            return monday_morning, UrgencyLevel.LOW.value

        elif key == 'eom':
            # Last day of current month
            last_day = calendar.monthrange(ref.year, ref.month)[1]
            eom = ref.replace(
                day=last_day,
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            return eom, UrgencyLevel.LOW.value

        elif key == 'eoq':
            # End of current quarter
            current_quarter = (ref.month - 1) // 3
            quarter_end_month = (current_quarter + 1) * 3
            last_day = calendar.monthrange(ref.year, quarter_end_month)[1]
            eoq = ref.replace(
                month=quarter_end_month,
                day=last_day,
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            return eoq, UrgencyLevel.MINIMAL.value

        elif key == 'eoy':
            eoy = ref.replace(
                month=12, day=31,
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            return eoy, UrgencyLevel.MINIMAL.value

        elif key == 'in_hours':
            hours = int(match.group(1))
            deadline = ref + timedelta(hours=hours)
            urgency = UrgencyLevel.HIGH.value if hours <= 4 else UrgencyLevel.MEDIUM.value
            return deadline, urgency

        elif key == 'in_days':
            days = int(match.group(1))
            deadline = ref + timedelta(days=days)
            deadline = deadline.replace(
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            if days <= 1:
                urgency = UrgencyLevel.HIGH.value
            elif days <= 3:
                urgency = UrgencyLevel.MEDIUM_HIGH.value
            else:
                urgency = UrgencyLevel.MEDIUM.value
            return deadline, urgency

        elif key == 'in_weeks':
            weeks = int(match.group(1))
            deadline = ref + timedelta(weeks=weeks)
            deadline = deadline.replace(
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            return deadline, UrgencyLevel.LOW.value

        elif key == 'end_of_sprint':
            # Assume 2-week sprint ending Friday
            days_until_friday = (4 - ref.weekday()) % 7
            if days_until_friday == 0 and ref.hour >= self.config.eod_time.hour:
                days_until_friday = 7
            friday = ref + timedelta(days=days_until_friday)
            friday_eod = friday.replace(
                hour=self.config.eod_time.hour,
                minute=self.config.eod_time.minute,
                second=0, microsecond=0
            )
            return friday_eod, UrgencyLevel.MEDIUM.value

        elif key == 'next_sprint':
            # 2 weeks from now
            deadline = ref + timedelta(weeks=2)
            return deadline, UrgencyLevel.LOW.value

        return None, UrgencyLevel.MINIMAL.value

    def _get_deadline_type(self, key: str) -> DeadlineType:
        """Map deadline key to DeadlineType."""
        if key in ('eod', 'eob', 'cob', 'eow', 'eom', 'eoq', 'eoy'):
            return DeadlineType.ACRONYM
        elif key in ('today', 'tonight', 'tomorrow', 'tomorrow_morning', 'tomorrow_afternoon'):
            return DeadlineType.RELATIVE_DAY
        elif key in ('this_week', 'next_week', 'beginning_next_week', 'end_of_sprint', 'next_sprint'):
            return DeadlineType.RELATIVE_PERIOD
        elif key.startswith('in_'):
            return DeadlineType.RELATIVE_PERIOD
        else:
            return DeadlineType.PERIOD_END

    def _extract_day_references(
        self,
        text: str,
        ref: datetime
    ) -> list[Deadline]:
        """Extract day name references (Monday, Friday, etc.)."""
        deadlines = []

        for pattern, _ in DAY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                day_name = match.group(1).lower()
                if day_name in DAY_NAMES:
                    target_weekday = DAY_NAMES[day_name]
                    days_ahead = (target_weekday - ref.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7  # Next occurrence

                    target_date = ref + timedelta(days=days_ahead)
                    target_date = target_date.replace(
                        hour=self.config.eod_time.hour,
                        minute=self.config.eod_time.minute,
                        second=0, microsecond=0
                    )

                    # Urgency based on days away
                    if days_ahead <= 2:
                        urgency = UrgencyLevel.MEDIUM_HIGH.value
                    elif days_ahead <= 5:
                        urgency = UrgencyLevel.MEDIUM.value
                    else:
                        urgency = UrgencyLevel.LOW.value

                    deadlines.append(Deadline(
                        text=match.group(0),
                        deadline_type=DeadlineType.DAY_NAME,
                        resolved_date=target_date,
                        urgency=urgency,
                        confidence=0.85,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    ))

        return deadlines

    def _extract_explicit_dates(
        self,
        text: str,
        ref: datetime
    ) -> list[Deadline]:
        """Extract explicit date patterns (12/25, Jan 15, etc.)."""
        deadlines = []

        for pattern, date_type in EXPLICIT_DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                resolved = self._resolve_explicit_date(date_type, match, ref)
                if resolved:
                    # Calculate urgency based on days until
                    days_until = (resolved - ref).days
                    if days_until <= 1:
                        urgency = UrgencyLevel.HIGH.value
                    elif days_until <= 3:
                        urgency = UrgencyLevel.MEDIUM_HIGH.value
                    elif days_until <= 7:
                        urgency = UrgencyLevel.MEDIUM.value
                    elif days_until <= 30:
                        urgency = UrgencyLevel.LOW.value
                    else:
                        urgency = UrgencyLevel.MINIMAL.value

                    deadlines.append(Deadline(
                        text=match.group(0),
                        deadline_type=DeadlineType.EXPLICIT_DATE,
                        resolved_date=resolved,
                        urgency=urgency,
                        confidence=0.95,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    ))

        return deadlines

    def _resolve_explicit_date(
        self,
        date_type: str,
        match: re.Match,
        ref: datetime
    ) -> Optional[datetime]:
        """Resolve explicit date match to datetime."""
        try:
            if date_type == 'slash_date':
                month = int(match.group(1))
                day = int(match.group(2))
                year = ref.year
                target = datetime(year, month, day,
                                  self.config.eod_time.hour,
                                  self.config.eod_time.minute)
                if target < ref:
                    target = target.replace(year=year + 1)
                return target

            elif date_type == 'slash_date_year':
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))
                if year < 100:
                    year = 2000 + year
                return datetime(year, month, day,
                               self.config.eod_time.hour,
                               self.config.eod_time.minute)

            elif date_type == 'dash_date':
                month = int(match.group(1))
                day = int(match.group(2))
                year = ref.year
                target = datetime(year, month, day,
                                  self.config.eod_time.hour,
                                  self.config.eod_time.minute)
                if target < ref:
                    target = target.replace(year=year + 1)
                return target

            elif date_type == 'dash_date_year':
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))
                if year < 100:
                    year = 2000 + year
                return datetime(year, month, day,
                               self.config.eod_time.hour,
                               self.config.eod_time.minute)

            elif date_type == 'iso_date':
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                return datetime(year, month, day,
                               self.config.eod_time.hour,
                               self.config.eod_time.minute)

            elif date_type == 'month_day':
                month_str = match.group(1).lower()[:3]
                day = int(match.group(2))
                month = MONTH_NAMES.get(month_str, 1)
                year = ref.year
                target = datetime(year, month, day,
                                  self.config.eod_time.hour,
                                  self.config.eod_time.minute)
                if target < ref:
                    target = target.replace(year=year + 1)
                return target

            elif date_type == 'day_month':
                day = int(match.group(1))
                month_str = match.group(2).lower()[:3]
                month = MONTH_NAMES.get(month_str, 1)
                year = ref.year
                target = datetime(year, month, day,
                                  self.config.eod_time.hour,
                                  self.config.eod_time.minute)
                if target < ref:
                    target = target.replace(year=year + 1)
                return target

        except ValueError:
            return None

        return None

    def _extract_times(
        self,
        text: str,
        ref: datetime
    ) -> list[Deadline]:
        """Extract time expressions (3pm, 15:00, etc.)."""
        deadlines = []

        for pattern, time_type in TIME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                resolved = self._resolve_time(time_type, match, ref)
                if resolved:
                    # Time deadlines are usually same-day urgent
                    hours_until = (resolved - ref).total_seconds() / 3600
                    if hours_until <= 2:
                        urgency = UrgencyLevel.CRITICAL.value
                    elif hours_until <= 4:
                        urgency = UrgencyLevel.HIGH.value
                    else:
                        urgency = UrgencyLevel.MEDIUM_HIGH.value

                    deadlines.append(Deadline(
                        text=match.group(0),
                        deadline_type=DeadlineType.EXPLICIT_TIME,
                        resolved_date=resolved,
                        urgency=urgency,
                        confidence=0.9,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    ))

        return deadlines

    def _resolve_time(
        self,
        time_type: str,
        match: re.Match,
        ref: datetime
    ) -> Optional[datetime]:
        """Resolve time match to datetime."""
        try:
            if time_type == 'time_12h':
                hour = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
                ampm = match.group(3).lower()
                if ampm == 'pm' and hour != 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
                target = ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= ref:
                    target += timedelta(days=1)
                return target

            elif time_type == 'time_24h':
                hour = int(match.group(1))
                minute = int(match.group(2))
                target = ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= ref:
                    target += timedelta(days=1)
                return target

            elif time_type == 'time_simple':
                hour = int(match.group(1))
                ampm = match.group(2).lower()
                if ampm == 'pm' and hour != 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
                target = ref.replace(hour=hour, minute=0, second=0, microsecond=0)
                if target <= ref:
                    target += timedelta(days=1)
                return target

        except ValueError:
            return None

        return None

    def _extract_llm_deadlines(
        self,
        text: str,
        ref: datetime
    ) -> list[Deadline]:
        """Use LLM for semantic deadline extraction.

        This finds deadline-related sentences that may not match explicit patterns.
        """
        if not self._llm_model or not self._deadline_embeddings:
            return []

        deadlines = []

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        import numpy as np

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 500:
                continue

            # Get embedding for this sentence
            sentence_embedding = self._llm_model.encode([sentence], convert_to_numpy=True)

            # Compute similarity with deadline phrases
            similarities = np.dot(self._deadline_embeddings, sentence_embedding.T).flatten()
            max_similarity = float(np.max(similarities))

            if max_similarity >= self.config.llm_threshold:
                # Extract any date-like tokens from the sentence
                resolved = None

                # Try to find a date in the sentence
                for pattern, date_type in EXPLICIT_DATE_PATTERNS:
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        resolved = self._resolve_explicit_date(date_type, match, ref)
                        break

                # Check for day names
                if not resolved:
                    for day_name, weekday in DAY_NAMES.items():
                        if day_name in sentence.lower():
                            days_ahead = (weekday - ref.weekday()) % 7
                            if days_ahead == 0:
                                days_ahead = 7
                            resolved = ref + timedelta(days=days_ahead)
                            resolved = resolved.replace(
                                hour=self.config.eod_time.hour,
                                minute=self.config.eod_time.minute,
                                second=0, microsecond=0
                            )
                            break

                # Find position in original text
                start_pos = text.find(sentence)
                end_pos = start_pos + len(sentence) if start_pos >= 0 else 0

                deadlines.append(Deadline(
                    text=sentence[:100] + ("..." if len(sentence) > 100 else ""),
                    deadline_type=DeadlineType.CONTEXTUAL,
                    resolved_date=resolved,
                    urgency=max_similarity * 0.6,  # Scale down LLM urgency
                    confidence=max_similarity,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    source="llm",
                ))

        return deadlines

    def _deduplicate_deadlines(self, deadlines: list[Deadline]) -> list[Deadline]:
        """Remove duplicate/overlapping deadlines, keeping highest confidence."""
        if not deadlines:
            return deadlines

        # Sort by position, then by confidence (highest first)
        sorted_deadlines = sorted(
            deadlines,
            key=lambda d: (d.start_pos, -d.confidence)
        )

        result = []
        last_end = -1

        for deadline in sorted_deadlines:
            # If this deadline doesn't overlap with previous, add it
            if deadline.start_pos >= last_end:
                result.append(deadline)
                last_end = deadline.end_pos
            # If it overlaps but has significantly higher confidence, replace
            elif result and deadline.confidence > result[-1].confidence + 0.2:
                result[-1] = deadline
                last_end = deadline.end_pos

        return result

    def get_primary_deadline(
        self,
        text: str,
        reference_date: Optional[datetime] = None
    ) -> Optional[Deadline]:
        """Get the most important deadline from text.

        Returns the deadline with highest urgency, or None if no deadlines found.
        """
        deadlines = self.extract(text, reference_date)
        return deadlines[0] if deadlines else None

    def get_resolved_deadlines(
        self,
        text: str,
        reference_date: Optional[datetime] = None
    ) -> list[Deadline]:
        """Get only deadlines that resolved to actual datetimes."""
        deadlines = self.extract(text, reference_date)
        return [d for d in deadlines if d.resolved_date is not None]


def extract_deadline(
    text: str,
    reference_date: Optional[datetime] = None,
    use_llm: bool = True
) -> Optional[Deadline]:
    """Convenience function to extract primary deadline from text.

    Args:
        text: Text to extract deadline from
        reference_date: Reference date for relative calculations
        use_llm: Whether to use LLM extraction

    Returns:
        Primary Deadline or None if no deadline found
    """
    config = DeadlineParserConfig(use_llm=use_llm)
    parser = DeadlineParser(config)
    return parser.get_primary_deadline(text, reference_date)


def extract_all_deadlines(
    text: str,
    reference_date: Optional[datetime] = None,
    use_llm: bool = True
) -> list[Deadline]:
    """Convenience function to extract all deadlines from text.

    Args:
        text: Text to extract deadlines from
        reference_date: Reference date for relative calculations
        use_llm: Whether to use LLM extraction

    Returns:
        List of Deadline objects sorted by urgency
    """
    config = DeadlineParserConfig(use_llm=use_llm)
    parser = DeadlineParser(config)
    return parser.extract(text, reference_date)


if __name__ == '__main__':
    # Example usage
    sample_emails = [
        """
        Hi Team,

        Please review the attached proposal and provide your feedback by EOD Friday.
        This is blocking the final presentation to the board.

        Can you also send me the updated spreadsheet? I need it ASAP.

        Thanks,
        Mike
        """,
        """
        Subject: Action Required - Q2 Report Due Jan 15

        The quarterly report needs to be submitted by January 15th.
        Please have your sections complete by next Wednesday.

        Time-sensitive: The CFO needs the executive summary by 3pm tomorrow.
        """,
        """
        Quick reminder: the client meeting is happening in 2 hours.
        Please review the materials before then.

        Also, I need the contract revisions by end of week.
        """,
    ]

    parser = DeadlineParser(DeadlineParserConfig(use_llm=False))

    for i, email in enumerate(sample_emails, 1):
        print(f"\n{'='*60}")
        print(f"Email {i}:")
        print(f"{'='*60}")

        deadlines = parser.extract(email)

        if deadlines:
            print(f"\nFound {len(deadlines)} deadline(s):")
            for d in deadlines:
                print(f"\n  Text: '{d.text}'")
                print(f"  Type: {d.deadline_type.value}")
                print(f"  Resolved: {d.resolved_date}")
                print(f"  Urgency: {d.urgency:.2f}")
                print(f"  Confidence: {d.confidence:.2f}")
                if d.resolved_date:
                    days = d.days_until()
                    if days is not None:
                        print(f"  Days until: {days:.1f}")
        else:
            print("\n  No deadlines found")
