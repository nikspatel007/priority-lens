"""Tests for Gmail rate limiter."""

from __future__ import annotations

import asyncio
from time import monotonic

import pytest

from priority_lens.integrations.gmail.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_default_requests_per_second(self) -> None:
        """Test default rate is 10 requests per second."""
        limiter = RateLimiter()
        assert limiter.requests_per_second == 10

    def test_custom_requests_per_second(self) -> None:
        """Test custom rate setting."""
        limiter = RateLimiter(requests_per_second=5)
        assert limiter.requests_per_second == 5

    def test_interval_calculation(self) -> None:
        """Test interval is correctly calculated."""
        limiter = RateLimiter(requests_per_second=10)
        assert limiter.interval_seconds == 0.1

        limiter2 = RateLimiter(requests_per_second=5)
        assert limiter2.interval_seconds == 0.2

    def test_reset(self) -> None:
        """Test reset clears the last request time."""
        limiter = RateLimiter()
        limiter._last_request_time = 100.0
        limiter.reset()
        assert limiter._last_request_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_first_request_immediate(self) -> None:
        """Test first request is immediate."""
        limiter = RateLimiter(requests_per_second=10)
        start = monotonic()
        await limiter.acquire()
        elapsed = monotonic() - start

        # First request should be nearly immediate
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_acquire_respects_rate_limit(self) -> None:
        """Test that acquire respects rate limit."""
        limiter = RateLimiter(requests_per_second=10)  # 0.1s interval

        start = monotonic()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = monotonic() - start

        # Second request should wait ~0.1s
        assert elapsed >= 0.08

    @pytest.mark.asyncio
    async def test_multiple_acquires_throttled(self) -> None:
        """Test multiple acquires are properly throttled."""
        limiter = RateLimiter(requests_per_second=20)  # 0.05s interval

        start = monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = monotonic() - start

        # 5 requests at 20/s = 4 intervals of 0.05s = 0.2s minimum
        assert elapsed >= 0.15

    @pytest.mark.asyncio
    async def test_acquire_after_reset(self) -> None:
        """Test that acquire is immediate after reset."""
        limiter = RateLimiter(requests_per_second=10)

        await limiter.acquire()
        limiter.reset()

        start = monotonic()
        await limiter.acquire()
        elapsed = monotonic() - start

        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self) -> None:
        """Test concurrent acquires are serialized."""
        limiter = RateLimiter(requests_per_second=10)

        async def acquire_and_count() -> float:
            await limiter.acquire()
            return monotonic()

        times = await asyncio.gather(
            acquire_and_count(),
            acquire_and_count(),
            acquire_and_count(),
        )
        times_sorted = sorted(times)

        # Each request should be spaced by at least interval
        for i in range(1, len(times_sorted)):
            diff = times_sorted[i] - times_sorted[i - 1]
            assert diff >= 0.08  # Allow small tolerance

    @pytest.mark.asyncio
    async def test_slow_requests_dont_throttle(self) -> None:
        """Test that slow gaps between requests don't cause waiting."""
        limiter = RateLimiter(requests_per_second=10)  # 0.1s interval

        await limiter.acquire()
        await asyncio.sleep(0.2)  # Wait longer than interval

        start = monotonic()
        await limiter.acquire()
        elapsed = monotonic() - start

        # Should be immediate since we waited longer than interval
        assert elapsed < 0.05

    def test_interval_for_one_per_second(self) -> None:
        """Test interval is 1 second for 1 request per second."""
        limiter = RateLimiter(requests_per_second=1)
        assert limiter.interval_seconds == 1.0

    @pytest.mark.asyncio
    async def test_high_rate_limit(self) -> None:
        """Test high rate limit works correctly."""
        limiter = RateLimiter(requests_per_second=100)

        start = monotonic()
        for _ in range(10):
            await limiter.acquire()
        elapsed = monotonic() - start

        # 10 requests at 100/s = 9 intervals of 0.01s = 0.09s minimum
        assert elapsed >= 0.08
        assert elapsed < 0.5  # But not too slow
