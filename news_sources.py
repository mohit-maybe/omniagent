"""Deprecated compatibility shim.

Use `ingestion.news_sources` for the canonical ingestion news sources.
"""

from __future__ import annotations

from ingestion.news_sources import *

__all__ = [
    "NewsSourceError",
    "RawArticle",
    "NewsSource",
    "MarketauxNewsSource",
    "FinnhubNewsSource",
]
