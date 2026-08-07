"""Deprecated compatibility shim.

Use the package-backed ingestion implementation in `ingestion.pipeline`
instead of the duplicate top-level module.
"""

from __future__ import annotations

from ingestion.pipeline import *

__all__ = [
    "TickerIngestionResult",
    "IngestionPipeline",
    "IngestionConfig",
    "MissingAPIKeyError",
    "PriceSourceError",
    "NewsSourceError",
    "RateLimitExceeded",
    "SentimentScorer",
    "get_default_scorer",
]
