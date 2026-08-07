"""
omnimarket.ingestion.news_sources
------------------------------------
Pluggable clients for financial news, normalized to
`models.NewsItem(ticker, headline, source, sentiment_score, published_at,
event_type)`. These sources return the RAW article (no sentiment_score
yet) - `pipeline.py` runs headlines through a SentimentScorer to fill
that in, so switching news providers and switching sentiment models are
independent decisions.

Sources implemented:
  - MarketauxNewsSource (primary - needs API key, free tier: 100 req/day)
  - FinnhubNewsSource   (fallback - needs API key, free tier: 60 calls/min)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from .config import IngestionConfig, MissingAPIKeyError, config as default_config


class NewsSourceError(RuntimeError):
    def __init__(self, source_name: str, ticker: str, detail: str):
        super().__init__(f"[{source_name}] failed to fetch news for {ticker}: {detail}")
        self.source_name = source_name
        self.ticker = ticker


@dataclass
class RawArticle:
    """An article before sentiment scoring - headline/body not yet turned into a score."""
    ticker: str
    headline: str
    source: str
    published_at: str
    event_type: str = "general"
    body_snippet: str = ""  # optional, if the provider returns one - improves scoring accuracy


_EVENT_KEYWORDS = {
    "earnings": ["earnings", "eps", "quarterly results", "revenue", "guidance"],
    "regulation": ["regulator", "sec ", "ftc", "antitrust", "lawsuit", "investigation", "fine", "sanction"],
    "macro": ["fed ", "federal reserve", "inflation", "cpi", "interest rate", "gdp", "unemployment"],
}


def _classify_event_type(headline: str) -> str:
    lower = headline.lower()
    for event_type, keywords in _EVENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return event_type
    return "general"


class NewsSource(ABC):
    name: str = "unknown"

    @abstractmethod
    def fetch_news(self, ticker: str, lookback_hours: float = 48.0, limit: int = 20) -> list[RawArticle]:
        raise NotImplementedError

    def _get(self, url: str, params: dict | None = None, headers: dict | None = None, timeout: float = 10.0):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise NewsSourceError(self.name, (params or {}).get("symbols", "?"), str(e)) from e


class MarketauxNewsSource(NewsSource):
    """
    News - primary source. Free tier: 100 requests/day.
    Docs: https://www.marketaux.com/documentation
    """
    name = "marketaux"

    def __init__(self, cfg: IngestionConfig | None = None):
        self.cfg = cfg or default_config

    def fetch_news(self, ticker: str, lookback_hours: float = 48.0, limit: int = 20) -> list[RawArticle]:
        if not self.cfg.marketaux_api_key:
            raise MissingAPIKeyError("Marketaux", "MARKETAUX_API_KEY")

        published_after = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M")

        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "symbols": ticker,
            "filter_entities": "true",
            "published_after": published_after,
            "limit": limit,
            "api_token": self.cfg.marketaux_api_key,
        }

        data = self._get(url, params=params, timeout=self.cfg.request_timeout_seconds)

        if "error" in data:
            raise NewsSourceError(self.name, ticker, str(data["error"]))

        articles = data.get("data")
        if articles is None:
            raise NewsSourceError(self.name, ticker, f"unexpected response shape: {list(data.keys())}")

        return [
            RawArticle(
                ticker=ticker,
                headline=a.get("title", ""),
                source=a.get("source", "Marketaux"),
                published_at=a.get("published_at", datetime.now(timezone.utc).isoformat()),
                event_type=_classify_event_type(a.get("title", "")),
                body_snippet=a.get("description", "") or a.get("snippet", ""),
            )
            for a in articles
        ]


class FinnhubNewsSource(NewsSource):
    """
    News - fallback source. Free tier: 60 API calls/minute.
    Docs: https://finnhub.io/docs/api/company-news
    """
    name = "finnhub"

    def __init__(self, cfg: IngestionConfig | None = None):
        self.cfg = cfg or default_config

    def fetch_news(self, ticker: str, lookback_hours: float = 48.0, limit: int = 20) -> list[RawArticle]:
        if not self.cfg.finnhub_api_key:
            raise MissingAPIKeyError("Finnhub", "FINNHUB_API_KEY")

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=lookback_hours)

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start.strftime("%Y-%m-%d"),
            "to": now.strftime("%Y-%m-%d"),
            "token": self.cfg.finnhub_api_key,
        }

        data = self._get(url, params=params, timeout=self.cfg.request_timeout_seconds)

        if not isinstance(data, list):
            raise NewsSourceError(self.name, ticker, f"unexpected response shape: {data}")

        articles = []
        for a in data[:limit]:
            published_at = datetime.fromtimestamp(a.get("datetime", 0), tz=timezone.utc).isoformat()
            headline = a.get("headline", "")
            articles.append(
                RawArticle(
                    ticker=ticker,
                    headline=headline,
                    source=a.get("source", "Finnhub"),
                    published_at=published_at,
                    event_type=_classify_event_type(headline),
                    body_snippet=a.get("summary", ""),
                )
            )
        return articles
