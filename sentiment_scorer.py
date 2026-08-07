"""Deprecated compatibility shim.

Use `ingestion.sentiment_scorer` for the canonical ingestion sentiment scorer.
"""

from __future__ import annotations

from ingestion.sentiment_scorer import *

__all__ = [
    "SentimentScorer",
    "VaderSentimentScorer",
    "TransformerSentimentScorer",
    "get_default_scorer",
]
