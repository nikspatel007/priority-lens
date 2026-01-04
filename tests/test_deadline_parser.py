"""Tests for enhanced deadline parsing."""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from deadline_parser import (
    DeadlineParser,
    DeadlineParserConfig,
    Deadline,
    DeadlineType,
    UrgencyLevel,
    extract_deadline,
    extract_all_deadlines,
)


class TestDeadlineType:
    """Test DeadlineType enum."""

    def test_all_types_defined(self):
        """Verify all deadline types are defined."""
        expected = [
            'EXPLICIT_DATE', 'EXPLICIT_TIME', 'DAY_NAME', 'RELATIVE_DAY',
            'PERIOD_END', 'ACRONYM', 'URGENCY', 'RELATIVE_PERIOD', 'CONTEXTUAL'
        ]
        for name in expected:
            assert hasattr(DeadlineType, name)


class TestUrgencyLevel:
    """Test UrgencyLevel enum."""

    def test_urgency_ordering(self):
        """Verify urgency levels are properly ordered."""
        assert UrgencyLevel.CRITICAL.value > UrgencyLevel.HIGH.value
        assert UrgencyLevel.HIGH.value > UrgencyLevel.MEDIUM_HIGH.value
        assert UrgencyLevel.MEDIUM_HIGH.value > UrgencyLevel.MEDIUM.value
        assert UrgencyLevel.MEDIUM.value > UrgencyLevel.LOW.value
        assert UrgencyLevel.LOW.value > UrgencyLevel.MINIMAL.value


class TestDeadline:
    """Test Deadline dataclass."""

    def test_days_until(self):
        """Test days_until calculation."""
        ref = datetime(2026, 1, 4, 12, 0)
        deadline = Deadline(
            text="by Friday",
            deadline_type=DeadlineType.DAY_NAME,
            resolved_date=datetime(2026, 1, 10, 17, 0),
            urgency=0.5,
            confidence=0.9,
            start_pos=0,
            end_pos=9,
        )
        days = deadline.days_until(ref)
        assert days is not None
        assert 6.0 < days < 6.5  # About 6 days

    def test_days_until_no_resolved_date(self):
        """Test days_until with no resolved date."""
        deadline = Deadline(
            text="ASAP",
            deadline_type=DeadlineType.URGENCY,
            resolved_date=None,
            urgency=1.0,
            confidence=0.95,
            start_pos=0,
            end_pos=4,
        )
        assert deadline.days_until() is None


class TestDeadlineParserConfig:
    """Test parser configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeadlineParserConfig()
        assert config.business_start.hour == 9
        assert config.business_end.hour == 17
        assert config.eod_time.hour == 17
        assert config.use_llm is True

    def test_custom_config(self):
        """Test custom configuration."""
        from datetime import time
        config = DeadlineParserConfig(
            business_start=time(8, 0),
            business_end=time(18, 0),
            use_llm=False,
        )
        assert config.business_start.hour == 8
        assert config.business_end.hour == 18
        assert config.use_llm is False


class TestUrgencyKeywords:
    """Test urgency keyword extraction."""

    @pytest.fixture
    def parser(self):
        """Create parser with LLM disabled for faster tests."""
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    def test_asap(self, parser):
        """Test ASAP detection."""
        deadlines = parser.extract("Please send this ASAP")
        assert len(deadlines) >= 1
        asap = next((d for d in deadlines if 'asap' in d.text.lower()), None)
        assert asap is not None
        assert asap.urgency == UrgencyLevel.CRITICAL.value
        assert asap.deadline_type == DeadlineType.URGENCY

    def test_urgent(self, parser):
        """Test urgent keyword detection."""
        deadlines = parser.extract("This is urgent!")
        assert len(deadlines) >= 1
        urgent = next((d for d in deadlines if 'urgent' in d.text.lower()), None)
        assert urgent is not None
        assert urgent.urgency == UrgencyLevel.CRITICAL.value

    def test_immediately(self, parser):
        """Test immediately keyword detection."""
        deadlines = parser.extract("Need this immediately")
        assert len(deadlines) >= 1
        immed = next((d for d in deadlines if 'immediately' in d.text.lower()), None)
        assert immed is not None
        assert immed.urgency == UrgencyLevel.CRITICAL.value

    def test_time_sensitive(self, parser):
        """Test time-sensitive detection."""
        deadlines = parser.extract("This request is time-sensitive")
        assert len(deadlines) >= 1

    def test_multiple_urgency_keywords(self, parser):
        """Test multiple urgency keywords in same text."""
        deadlines = parser.extract("This is urgent and needs to be done ASAP")
        # Should detect both but may deduplicate
        assert len(deadlines) >= 1
        assert any(d.urgency == UrgencyLevel.CRITICAL.value for d in deadlines)


class TestImplicitDeadlines:
    """Test implicit deadline extraction (EOD, this week, etc.)."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    @pytest.fixture
    def ref_date(self):
        """Reference date: Monday at noon."""
        return datetime(2026, 1, 5, 12, 0, 0)  # Monday

    def test_eod(self, parser, ref_date):
        """Test EOD detection."""
        deadlines = parser.extract("Need this by EOD", ref_date)
        eod = next((d for d in deadlines if d.deadline_type == DeadlineType.ACRONYM), None)
        assert eod is not None
        assert eod.resolved_date.hour == 17
        assert eod.resolved_date.date() == ref_date.date()
        assert eod.urgency >= UrgencyLevel.HIGH.value

    def test_end_of_day(self, parser, ref_date):
        """Test 'end of day' detection."""
        deadlines = parser.extract("Please complete by end of day", ref_date)
        assert len(deadlines) >= 1
        eod = deadlines[0]
        assert eod.resolved_date.hour == 17

    def test_cob(self, parser, ref_date):
        """Test COB (close of business) detection."""
        deadlines = parser.extract("Submit before COB", ref_date)
        cob = next((d for d in deadlines if 'cob' in d.text.lower()), None)
        assert cob is not None
        assert cob.resolved_date.hour == 17

    def test_eow(self, parser, ref_date):
        """Test EOW (end of week) detection."""
        deadlines = parser.extract("Complete by EOW", ref_date)
        eow = next((d for d in deadlines if 'eow' in d.text.lower()), None)
        assert eow is not None
        # Should resolve to Friday
        assert eow.resolved_date.weekday() == 4  # Friday

    def test_this_week(self, parser, ref_date):
        """Test 'this week' detection."""
        deadlines = parser.extract("Need this done this week", ref_date)
        assert len(deadlines) >= 1
        tw = deadlines[0]
        assert tw.resolved_date.weekday() == 4  # Friday

    def test_eom(self, parser, ref_date):
        """Test EOM (end of month) detection."""
        deadlines = parser.extract("Due by EOM", ref_date)
        eom = next((d for d in deadlines if 'eom' in d.text.lower()), None)
        assert eom is not None
        # Should resolve to last day of January
        assert eom.resolved_date.day == 31
        assert eom.resolved_date.month == 1

    def test_eoq(self, parser, ref_date):
        """Test EOQ (end of quarter) detection."""
        deadlines = parser.extract("Finish by EOQ", ref_date)
        eoq = next((d for d in deadlines if 'eoq' in d.text.lower()), None)
        assert eoq is not None
        # Q1 ends March 31
        assert eoq.resolved_date.month == 3
        assert eoq.resolved_date.day == 31

    def test_today(self, parser, ref_date):
        """Test 'today' detection."""
        deadlines = parser.extract("Need this by today", ref_date)
        assert len(deadlines) >= 1
        today = deadlines[0]
        assert today.resolved_date.date() == ref_date.date()

    def test_tomorrow(self, parser, ref_date):
        """Test 'tomorrow' detection."""
        deadlines = parser.extract("Complete by tomorrow", ref_date)
        assert len(deadlines) >= 1
        tom = deadlines[0]
        assert tom.resolved_date.date() == (ref_date + timedelta(days=1)).date()

    def test_next_week(self, parser, ref_date):
        """Test 'next week' detection."""
        deadlines = parser.extract("Start next week", ref_date)
        assert len(deadlines) >= 1
        nw = deadlines[0]
        # Should be next Monday
        assert nw.resolved_date.weekday() == 0  # Monday
        assert nw.resolved_date > ref_date

    def test_in_hours(self, parser, ref_date):
        """Test 'in X hours' detection."""
        deadlines = parser.extract("Respond within 2 hours", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        expected = ref_date + timedelta(hours=2)
        assert abs((d.resolved_date - expected).total_seconds()) < 60

    def test_in_days(self, parser, ref_date):
        """Test 'in X days' detection."""
        deadlines = parser.extract("Complete in 3 days", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        assert d.resolved_date.date() == (ref_date + timedelta(days=3)).date()


class TestDayNameReferences:
    """Test day name reference extraction (Monday, Friday, etc.)."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    @pytest.fixture
    def ref_date(self):
        """Reference date: Wednesday."""
        return datetime(2026, 1, 7, 12, 0, 0)  # Wednesday

    def test_by_friday(self, parser, ref_date):
        """Test 'by Friday' detection."""
        deadlines = parser.extract("Submit by Friday", ref_date)
        assert len(deadlines) >= 1
        fri = deadlines[0]
        assert fri.resolved_date.weekday() == 4  # Friday
        # Should be this Friday (2 days from Wednesday)
        assert fri.resolved_date.date() == (ref_date + timedelta(days=2)).date()

    def test_before_monday(self, parser, ref_date):
        """Test 'before Monday' detection."""
        deadlines = parser.extract("Need this before Monday", ref_date)
        assert len(deadlines) >= 1
        mon = deadlines[0]
        assert mon.resolved_date.weekday() == 0  # Monday
        # Should be next Monday
        assert mon.resolved_date > ref_date

    def test_this_wednesday(self, parser, ref_date):
        """Test 'this Wednesday' when today is Wednesday."""
        deadlines = parser.extract("Meeting this Wednesday", ref_date)
        # Should find Wednesday, next occurrence since today
        if deadlines:
            wed = deadlines[0]
            assert wed.resolved_date.weekday() == 2

    def test_friday_morning(self, parser, ref_date):
        """Test 'Friday morning' detection."""
        deadlines = parser.extract("Call scheduled Friday morning", ref_date)
        assert len(deadlines) >= 1
        fri = deadlines[0]
        assert fri.resolved_date.weekday() == 4


class TestExplicitDates:
    """Test explicit date extraction (12/25, Jan 15, etc.)."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    @pytest.fixture
    def ref_date(self):
        return datetime(2026, 1, 4, 12, 0, 0)

    def test_slash_date_mm_dd(self, parser, ref_date):
        """Test MM/DD format."""
        deadlines = parser.extract("Due 1/15", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        assert d.resolved_date.month == 1
        assert d.resolved_date.day == 15

    def test_slash_date_full(self, parser, ref_date):
        """Test MM/DD/YYYY format."""
        deadlines = parser.extract("Deadline: 3/15/2026", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        assert d.resolved_date.month == 3
        assert d.resolved_date.day == 15
        assert d.resolved_date.year == 2026

    def test_month_day(self, parser, ref_date):
        """Test 'January 15' format."""
        deadlines = parser.extract("Submit by January 15", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        assert d.resolved_date.month == 1
        assert d.resolved_date.day == 15

    def test_month_day_short(self, parser, ref_date):
        """Test 'Jan 15th' format."""
        deadlines = parser.extract("Due Jan 15th", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        assert d.resolved_date.month == 1
        assert d.resolved_date.day == 15

    def test_day_month(self, parser, ref_date):
        """Test '15th of January' format."""
        deadlines = parser.extract("Complete by 15th of January", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        assert d.resolved_date.month == 1
        assert d.resolved_date.day == 15

    def test_past_date_rolls_to_next_year(self, parser):
        """Test that past dates roll to next year."""
        ref = datetime(2026, 12, 1, 12, 0)
        deadlines = parser.extract("Due 1/15", ref)
        assert len(deadlines) >= 1
        d = deadlines[0]
        # January 15 is before December 1, so should be 2027
        assert d.resolved_date.year == 2027

    def test_iso_date(self, parser, ref_date):
        """Test YYYY-MM-DD format."""
        deadlines = parser.extract("Deadline: 2026-02-15", ref_date)
        assert len(deadlines) >= 1
        d = deadlines[0]
        assert d.resolved_date.year == 2026
        assert d.resolved_date.month == 2
        assert d.resolved_date.day == 15


class TestTimeExpressions:
    """Test time expression extraction (3pm, 15:00, etc.)."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    @pytest.fixture
    def ref_date(self):
        return datetime(2026, 1, 4, 10, 0, 0)  # 10 AM

    def test_time_12h(self, parser, ref_date):
        """Test 12-hour time format."""
        deadlines = parser.extract("Meeting at 3pm", ref_date)
        assert len(deadlines) >= 1
        t = deadlines[0]
        assert t.resolved_date.hour == 15
        assert t.resolved_date.minute == 0

    def test_time_12h_with_minutes(self, parser, ref_date):
        """Test 12-hour time with minutes."""
        deadlines = parser.extract("Call by 2:30 PM", ref_date)
        assert len(deadlines) >= 1
        t = deadlines[0]
        assert t.resolved_date.hour == 14
        assert t.resolved_date.minute == 30

    def test_time_am(self, parser, ref_date):
        """Test AM time."""
        deadlines = parser.extract("Submit before 9am", ref_date)
        # Since ref is 10am, 9am would be tomorrow
        assert len(deadlines) >= 1
        t = deadlines[0]
        assert t.resolved_date.hour == 9
        assert t.resolved_date.date() == (ref_date + timedelta(days=1)).date()

    def test_time_24h(self, parser, ref_date):
        """Test 24-hour time format."""
        deadlines = parser.extract("Deadline at 14:00", ref_date)
        assert len(deadlines) >= 1
        t = deadlines[0]
        assert t.resolved_date.hour == 14


class TestComplexEmails:
    """Test extraction from realistic email content."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    @pytest.fixture
    def ref_date(self):
        return datetime(2026, 1, 4, 12, 0, 0)

    def test_email_with_multiple_deadlines(self, parser, ref_date):
        """Test email with multiple different deadlines."""
        email = """
        Hi Team,

        Please review the attached proposal and provide your feedback by EOD Friday.
        This is blocking the final presentation to the board.

        Can you also send me the updated spreadsheet? I need it ASAP.

        The quarterly report is due by January 15th.

        Thanks,
        Mike
        """
        deadlines = parser.extract(email, ref_date)
        assert len(deadlines) >= 3

        # Should have ASAP (highest urgency)
        asap = next((d for d in deadlines if d.deadline_type == DeadlineType.URGENCY), None)
        assert asap is not None

        # Should have EOD
        eod = next((d for d in deadlines if 'eod' in d.text.lower()), None)
        assert eod is not None

        # Sorted by urgency - ASAP should be first
        assert deadlines[0].urgency >= deadlines[-1].urgency

    def test_email_with_nested_context(self, parser, ref_date):
        """Test email with deadlines in various contexts."""
        email = """
        Subject: Action Required - Q2 Report Due Jan 15

        The quarterly report needs to be submitted by January 15th.
        Please have your sections complete by next Wednesday.

        Time-sensitive: The CFO needs the executive summary by 3pm tomorrow.
        """
        deadlines = parser.extract(email, ref_date)

        # Should find multiple deadlines
        assert len(deadlines) >= 2

        # Should find the time-sensitive marker
        assert any(d.deadline_type == DeadlineType.URGENCY for d in deadlines)

    def test_email_with_no_deadlines(self, parser, ref_date):
        """Test email with no deadlines."""
        email = """
        Hi,

        Just wanted to say thanks for your help last week.
        The project went well and the client was happy.

        Best,
        Sarah
        """
        deadlines = parser.extract(email, ref_date)
        assert len(deadlines) == 0


class TestPrimaryDeadline:
    """Test primary deadline extraction."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    def test_get_primary_deadline(self, parser):
        """Test getting the most urgent deadline."""
        email = "Need this ASAP, but definitely by Friday"
        primary = parser.get_primary_deadline(email)

        assert primary is not None
        # ASAP should be primary (most urgent)
        assert primary.urgency >= 0.9

    def test_get_primary_no_deadline(self, parser):
        """Test primary deadline when none exist."""
        email = "Thanks for the update!"
        primary = parser.get_primary_deadline(email)
        assert primary is None

    def test_get_resolved_deadlines(self, parser):
        """Test getting only resolved deadlines."""
        ref = datetime(2026, 1, 4, 12, 0)
        email = "ASAP! Also by Friday please."
        resolved = parser.get_resolved_deadlines(email, ref)

        # ASAP doesn't resolve to a date, Friday does
        assert all(d.resolved_date is not None for d in resolved)
        assert any(d.resolved_date.weekday() == 4 for d in resolved)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_extract_deadline(self):
        """Test extract_deadline convenience function."""
        deadline = extract_deadline("Submit by EOD", use_llm=False)
        assert deadline is not None
        assert deadline.deadline_type == DeadlineType.ACRONYM

    def test_extract_all_deadlines(self):
        """Test extract_all_deadlines convenience function."""
        deadlines = extract_all_deadlines(
            "ASAP! Also by Friday and Jan 15",
            use_llm=False
        )
        assert len(deadlines) >= 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    def test_empty_string(self, parser):
        """Test empty input."""
        deadlines = parser.extract("")
        assert deadlines == []

    def test_very_long_text(self, parser):
        """Test very long input."""
        long_text = "Some text " * 10000 + " by EOD"
        deadlines = parser.extract(long_text)
        assert len(deadlines) >= 1

    def test_special_characters(self, parser):
        """Test text with special characters."""
        text = "Due by 1/15!!! Please submit ASAP??? <urgent>"
        deadlines = parser.extract(text)
        assert len(deadlines) >= 1

    def test_case_insensitivity(self, parser):
        """Test case insensitivity."""
        texts = ["BY EOD", "by eod", "By Eod", "BY eod"]
        for text in texts:
            deadlines = parser.extract(text)
            assert len(deadlines) >= 1

    def test_friday_when_today_is_friday_after_eod(self, parser):
        """Test Friday reference when it's Friday after 5pm."""
        ref = datetime(2026, 1, 9, 18, 0)  # Friday 6pm
        deadlines = parser.extract("by Friday", ref)
        if deadlines:
            # Should be next Friday
            assert deadlines[0].resolved_date > ref

    def test_midnight_time(self, parser):
        """Test midnight time reference."""
        ref = datetime(2026, 1, 4, 12, 0)
        deadlines = parser.extract("Submit by 12am", ref)
        # 12am is midnight, should be next day
        if deadlines:
            assert deadlines[0].resolved_date.hour == 0


class TestUrgencyScoring:
    """Test urgency score calculations."""

    @pytest.fixture
    def parser(self):
        return DeadlineParser(DeadlineParserConfig(use_llm=False))

    @pytest.fixture
    def ref_date(self):
        return datetime(2026, 1, 4, 12, 0)

    def test_urgency_ordering(self, parser, ref_date):
        """Test that urgency scores are properly ordered."""
        emails = {
            'asap': "Do this ASAP",
            'today': "Complete by today",
            'tomorrow': "Due tomorrow",
            'this_week': "Finish this week",
            'next_week': "Start next week",
        }

        urgencies = {}
        for key, text in emails.items():
            deadlines = parser.extract(text, ref_date)
            if deadlines:
                urgencies[key] = deadlines[0].urgency

        # ASAP should be highest
        if 'asap' in urgencies and 'today' in urgencies:
            assert urgencies['asap'] >= urgencies['today']

        # Today should be higher than tomorrow
        if 'today' in urgencies and 'tomorrow' in urgencies:
            assert urgencies['today'] >= urgencies['tomorrow']

        # Tomorrow higher than this_week
        if 'tomorrow' in urgencies and 'this_week' in urgencies:
            assert urgencies['tomorrow'] >= urgencies['this_week']

    def test_explicit_date_urgency_by_proximity(self, parser, ref_date):
        """Test that closer dates have higher urgency."""
        d1 = parser.extract("Due 1/5", ref_date)  # Tomorrow
        d2 = parser.extract("Due 1/15", ref_date)  # 11 days away

        if d1 and d2:
            assert d1[0].urgency > d2[0].urgency
