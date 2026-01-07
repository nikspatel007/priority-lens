"""Rate limiting utilities for Gmail API."""

from __future__ import annotations

import asyncio
from time import monotonic


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Implements a sliding window rate limiter that ensures requests
    don't exceed a specified rate per second. This is essential for
    staying within Gmail API quotas.

    Gmail API has per-user quotas:
    - 25,000 queries per 100 seconds per user
    - We use a conservative default of 10 requests/second

    Attributes:
        requests_per_second: Maximum requests allowed per second.
    """

    def __init__(self, requests_per_second: int = 10) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second (default: 10).
        """
        self.requests_per_second = requests_per_second
        self._interval = 1.0 / requests_per_second
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request is allowed.

        This method blocks until the rate limit allows another request.
        It should be called before each API request.

        Example:
            limiter = RateLimiter(requests_per_second=10)
            await limiter.acquire()
            await make_api_request()
        """
        async with self._lock:
            now = monotonic()
            time_since_last = now - self._last_request_time
            wait_time = self._interval - time_since_last

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self._last_request_time = monotonic()

    def reset(self) -> None:
        """Reset the rate limiter.

        This clears the last request time, allowing immediate requests.
        Useful for testing or when switching between users.
        """
        self._last_request_time = 0.0

    @property
    def interval_seconds(self) -> float:
        """Get the interval between requests in seconds.

        Returns:
            Minimum seconds between consecutive requests.
        """
        return self._interval
