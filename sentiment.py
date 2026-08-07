"""
omnimarket.sentiment
----------------------
Aggregates raw NewsItem records (already sentiment-scored, e.g. by a
HuggingFace distilbert-sst2 pass upstream) into a per-ticker
SentimentSnapshot: mean score, article velocity, and score delta versus
the prior window. This is the "sentiment momentum" input the prediction
engine combines with technical indicators.

This module does NOT call any news API itself - inject NewsItem lists
from whatever ingestion job you wire up (Marketaux / APITube / Finnhub).
That keeps this layer testable and swappable.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from models import NewsItem, SentimentSnapshot


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class SentimentAggregator:
    def __init__(self, lookback_hours: float = 24.0, prior_window_hours: float = 24.0):
        self.lookback_hours = lookback_hours
        self.prior_window_hours = prior_window_hours

    def aggregate(self, news: Iterable[NewsItem], as_of: datetime | None = None) -> dict[str, SentimentSnapshot]:
        as_of = as_of or datetime.now(timezone.utc)
        window_start = as_of - timedelta(hours=self.lookback_hours)
        prior_start = window_start - timedelta(hours=self.prior_window_hours)

        by_ticker_current: dict[str, list[NewsItem]] = defaultdict(list)
        by_ticker_prior: dict[str, list[NewsItem]] = defaultdict(list)

        for item in news:
            ts = _parse_ts(item.published_at)
            if window_start <= ts <= as_of:
                by_ticker_current[item.ticker].append(item)
            elif prior_start <= ts < window_start:
                by_ticker_prior[item.ticker].append(item)

        snapshots: dict[str, SentimentSnapshot] = {}
        all_tickers = set(by_ticker_current) | set(by_ticker_prior)

        for ticker in all_tickers:
            current = by_ticker_current.get(ticker, [])
            prior = by_ticker_prior.get(ticker, [])

            mean_current = sum(a.sentiment_score for a in current) / len(current) if current else 0.0
            mean_prior = sum(a.sentiment_score for a in prior) / len(prior) if prior else mean_current

            velocity = len(current) / self.lookback_hours if self.lookback_hours else 0.0

            event_types = [a.event_type for a in current]
            dominant_event = max(set(event_types), key=event_types.count) if event_types else "general"

            snapshots[ticker] = SentimentSnapshot(
                ticker=ticker,
                mean_score=round(mean_current, 4),
                velocity=round(velocity, 4),
                score_delta=round(mean_current - mean_prior, 4),
                article_count=len(current),
                dominant_event_type=dominant_event,
            )

        return snapshots
