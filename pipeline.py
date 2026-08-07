"""
omnimarket.ingestion.pipeline
---------------------------------
Ties price sources + news sources + sentiment scoring together into the
exact inputs prediction_engine.py / indicators.py / sentiment.py expect,
with:
  - Primary/fallback source chains (Alpaca -> Alpha Vantage for stocks;
    Binance -> CoinCap for crypto; Marketaux -> Finnhub for news)
  - Rate limiting per provider, so a batch job over many tickers doesn't
    blow through a free-tier daily quota
  - Graceful degradation: a ticker with no news, or a source that's
    misconfigured/down, doesn't take the whole batch down - it's logged
    and skipped, and downstream code sees "no data for this ticker"
    rather than a crash job

This is the piece you point at real market hours and let run - either
manually, or via scheduler.py on a timer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from models import AssetClass, NewsItem
from config import IngestionConfig, MissingAPIKeyError, config as default_config
from price_sources import (
    AlpacaPriceSource, AlphaVantagePriceSource, BinancePriceSource, CoinCapPriceSource,
    PriceSourceError,
)
from news_sources import MarketauxNewsSource, FinnhubNewsSource, NewsSourceError, RawArticle
from rate_limiter import DailyQuotaLimiter, PerMinuteLimiter, RateLimitExceeded
from sentiment_scorer import SentimentScorer, get_default_scorer

logger = logging.getLogger("omnimarket.ingestion")


@dataclass
class TickerIngestionResult:
    ticker: str
    asset_class: AssetClass
    ohlcv: pd.DataFrame | None = None
    news: list[NewsItem] = field(default_factory=list)
    price_source_used: str | None = None
    news_source_used: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.ohlcv is not None and not self.ohlcv.empty


class IngestionPipeline:
    def __init__(self, cfg: IngestionConfig | None = None, sentiment_scorer: SentimentScorer | None = None):
        self.cfg = cfg or default_config
        self.scorer = sentiment_scorer or get_default_scorer()

        self._stock_price_chain = [AlpacaPriceSource(self.cfg), AlphaVantagePriceSource(self.cfg)]
        self._crypto_price_chain = [BinancePriceSource(self.cfg), CoinCapPriceSource(self.cfg)]
        self._news_chain = [MarketauxNewsSource(self.cfg), FinnhubNewsSource(self.cfg)]

        self._marketaux_limiter = DailyQuotaLimiter("marketaux", self.cfg.marketaux_daily_limit)
        self._finnhub_limiter = PerMinuteLimiter("finnhub", self.cfg.finnhub_per_minute_limit)
        self._alpha_vantage_limiter = PerMinuteLimiter("alpha_vantage", self.cfg.alpha_vantage_per_minute_limit)

    # ---------------- price ----------------

    def _price_chain_for(self, asset_class: AssetClass):
        return self._stock_price_chain if asset_class == AssetClass.STOCK else self._crypto_price_chain

    def fetch_price(self, ticker: str, asset_class: AssetClass, lookback_bars: int = 90) -> tuple[pd.DataFrame | None, str | None, list[str]]:
        errors = []
        for source in self._price_chain_for(asset_class):
            try:
                if source.name == "alpha_vantage":
                    self._alpha_vantage_limiter.consume()
                df = source.fetch_ohlcv(ticker, lookback_bars=lookback_bars)
                if df is not None and not df.empty:
                    return df, source.name, errors
                errors.append(f"{source.name}: returned empty data")
            except MissingAPIKeyError as e:
                errors.append(f"{source.name}: {e}")
            except RateLimitExceeded as e:
                errors.append(f"{source.name}: {e}")
            except PriceSourceError as e:
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001 - one bad source should never kill the batch
                errors.append(f"{source.name}: unexpected error: {e}")
        return None, None, errors

    # ---------------- news ----------------

    def fetch_news(self, ticker: str, lookback_hours: float = 48.0) -> tuple[list[NewsItem], str | None, list[str]]:
        errors = []
        for source in self._news_chain:
            try:
                if source.name == "marketaux":
                    self._marketaux_limiter.consume()
                elif source.name == "finnhub":
                    self._finnhub_limiter.consume()

                raw_articles: list[RawArticle] = source.fetch_news(ticker, lookback_hours=lookback_hours)
                news_items = [self._score_article(a) for a in raw_articles]
                return news_items, source.name, errors
            except MissingAPIKeyError as e:
                errors.append(f"{source.name}: {e}")
            except RateLimitExceeded as e:
                errors.append(f"{source.name}: {e}")
            except NewsSourceError as e:
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{source.name}: unexpected error: {e}")
        return [], None, errors

    def _score_article(self, article: RawArticle) -> NewsItem:
        # Score on headline + snippet together when available - more context,
        # better signal, especially for short/ambiguous headlines.
        text = f"{article.headline}. {article.body_snippet}".strip()
        score = self.scorer.score(text)
        return NewsItem(
            ticker=article.ticker,
            headline=article.headline,
            source=article.source,
            sentiment_score=round(score, 4),
            published_at=article.published_at,
            event_type=article.event_type,
        )

    # ---------------- combined ----------------

    def ingest_ticker(self, ticker: str, asset_class: AssetClass, lookback_bars: int = 90,
                       news_lookback_hours: float = 48.0) -> TickerIngestionResult:
        result = TickerIngestionResult(ticker=ticker, asset_class=asset_class)

        ohlcv, price_source, price_errors = self.fetch_price(ticker, asset_class, lookback_bars)
        result.ohlcv = ohlcv
        result.price_source_used = price_source
        result.errors.extend(price_errors)

        news, news_source, news_errors = self.fetch_news(ticker, news_lookback_hours)
        result.news = news
        result.news_source_used = news_source
        result.errors.extend(news_errors)

        if not result.ok:
            logger.warning("No usable price data for %s. Errors: %s", ticker, price_errors)
        if not news:
            logger.info("No news retrieved for %s (non-fatal). Errors: %s", ticker, news_errors)

        return result

    def ingest_batch(self, tickers: dict[str, AssetClass], lookback_bars: int = 90,
                      news_lookback_hours: float = 48.0) -> dict[str, TickerIngestionResult]:
        results = {}
        for ticker, asset_class in tickers.items():
            results[ticker] = self.ingest_ticker(ticker, asset_class, lookback_bars, news_lookback_hours)
        return results
