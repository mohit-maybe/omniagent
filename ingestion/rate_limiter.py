"""
omnimarket.ingestion.rate_limiter
------------------------------------
Small, dependency-free rate limiters so ingestion respects each provider's
free-tier quota instead of getting the API key throttled or banned.

Two shapes covered by the spec's free tiers:
  - DailyQuotaLimiter: N requests per rolling 24h (Marketaux: 100/day)
  - PerMinuteLimiter: N requests per rolling 60s (Finnhub: 60/min, Alpha
    Vantage: 5/min)

Both raise RateLimitExceeded rather than silently blocking/sleeping, so
the caller (pipeline.py) can decide whether to wait, skip, or fall back
to a secondary source - blocking silently inside a rate limiter tends to
hide real problems in a scheduled job.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


class RateLimitExceeded(RuntimeError):
    def __init__(self, source_name: str, limit: int, window_desc: str):
        super().__init__(f"{source_name} rate limit exceeded: {limit} requests per {window_desc}")


@dataclass
class _WindowLimiter:
    source_name: str
    limit: int
    window_seconds: float
    window_desc: str
    _timestamps: deque = field(default_factory=deque)

    def check(self) -> None:
        """Raise RateLimitExceeded if calling now would exceed the quota. Does not consume."""
        now = time.time()
        self._evict(now)
        if len(self._timestamps) >= self.limit:
            raise RateLimitExceeded(self.source_name, self.limit, self.window_desc)

    def consume(self) -> None:
        """Record a request. Call this right before actually making the HTTP call."""
        now = time.time()
        self._evict(now)
        if len(self._timestamps) >= self.limit:
            raise RateLimitExceeded(self.source_name, self.limit, self.window_desc)
        self._timestamps.append(now)

    def remaining(self) -> int:
        self._evict(time.time())
        return max(0, self.limit - len(self._timestamps))

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


def DailyQuotaLimiter(source_name: str, limit: int) -> _WindowLimiter:
    return _WindowLimiter(source_name=source_name, limit=limit, window_seconds=86400, window_desc="24h")


def PerMinuteLimiter(source_name: str, limit: int) -> _WindowLimiter:
    return _WindowLimiter(source_name=source_name, limit=limit, window_seconds=60, window_desc="60s")
