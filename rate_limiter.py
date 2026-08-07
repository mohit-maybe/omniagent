"""Deprecated compatibility shim.

Use `ingestion.rate_limiter` for the canonical ingestion rate limiters.
"""

from __future__ import annotations

from ingestion.rate_limiter import *

__all__ = [
    "RateLimitExceeded",
    "DailyQuotaLimiter",
    "PerMinuteLimiter",
]
