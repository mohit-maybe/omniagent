"""
omnimarket.models
------------------
Core data structures shared across the prediction engine, risk manager,
and paper-trading engine. Kept dependency-light (stdlib dataclasses + enum)
so this module can be imported by the API layer, batch jobs, or tests
without pulling in pandas/numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"
    RANGING = "ranging"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OHLCVBar:
    """Single candle of price data."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class NewsItem:
    """A single ingested news article, pre-scored by a sentiment model."""
    ticker: str
    headline: str
    source: str
    sentiment_score: float  # -1.0 (bearish) to +1.0 (bullish)
    published_at: str
    event_type: str = "general"  # earnings, regulation, macro, general...


@dataclass
class TechnicalSnapshot:
    """Output of the indicator layer for one ticker at one point in time."""
    ticker: str
    rsi: float
    macd: float
    macd_signal: float
    sma_fast: float
    sma_slow: float
    atr: float
    last_price: float


@dataclass
class SentimentSnapshot:
    """Aggregated sentiment for one ticker over the lookback window."""
    ticker: str
    mean_score: float
    velocity: float          # rate of new articles per hour
    score_delta: float       # change in mean score vs prior window
    article_count: int
    dominant_event_type: str = "general"


@dataclass
class Signal:
    """A single trade signal produced by the PredictionEngine."""
    ticker: str
    asset_class: AssetClass
    direction: Direction
    confidence: float               # 0-100
    regime: MarketRegime
    rationale: str
    components: dict = field(default_factory=dict)  # {"sentiment": float, "technical": float, "regime": float} in [-1,1]
    generated_at: str = field(default_factory=_now)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class RiskEvent:
    """A flagged high-risk condition that can block or throttle trading."""
    ticker: Optional[str]
    reason: str
    severity: str = "high"  # low, medium, high
    detected_at: str = field(default_factory=_now)


@dataclass
class Position:
    ticker: str
    direction: Direction
    entry_price: float
    size_usd: float
    stop_loss: float
    take_profit: float
    opened_at: str = field(default_factory=_now)


@dataclass
class Trade:
    """A closed or open paper trade, fully logged for auditability."""
    trade_id: str
    ticker: str
    asset_class: AssetClass
    direction: Direction
    entry_price: float
    size_usd: float
    stop_loss: float
    take_profit: float
    rationale: str
    confidence: float
    opened_at: str
    signal_components: dict = field(default_factory=dict)  # copied from the originating Signal
    status: str = "open"          # open, closed_win, closed_loss, closed_manual
    exit_price: Optional[float] = None
    closed_at: Optional[str] = None
    pnl_usd: Optional[float] = None
    roi_pct: Optional[float] = None


@dataclass
class PortfolioState:
    cash_usd: float
    starting_equity: float
    positions: list = field(default_factory=list)   # list[Position]
    trade_log: list = field(default_factory=list)    # list[Trade]

    @property
    def equity(self) -> float:
        invested = sum(p.size_usd for p in self.positions)
        return self.cash_usd + invested

    @property
    def total_return_pct(self) -> float:
        return (self.equity - self.starting_equity) / self.starting_equity * 100

    @property
    def exposure_pct(self) -> float:
        invested = sum(p.size_usd for p in self.positions)
        return invested / self.starting_equity * 100
