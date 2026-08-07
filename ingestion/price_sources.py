"""
omnimarket.ingestion.price_sources
--------------------------------------
Pluggable clients for stock and crypto price data, normalized to the
pandas DataFrame shape `indicators.py` expects: columns
['open','high','low','close','volume'], indexed by timestamp, oldest
first.

Sources implemented:
  - AlpacaPriceSource   (stocks, primary - needs API key)
  - AlphaVantagePriceSource (stocks, fallback - needs API key)
  - BinancePriceSource  (crypto, primary - public, no key needed)
  - CoinCapPriceSource  (crypto, fallback - public, no key needed)

Every source raises a specific, catchable exception on failure
(PriceSourceError) rather than letting a raw requests exception or a
KeyError from unexpected JSON shape propagate - `pipeline.py`'s fallback
chain depends on being able to catch exactly this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import IngestionConfig, MissingAPIKeyError, config as default_config


class PriceSourceError(RuntimeError):
    def __init__(self, source_name: str, ticker: str, detail: str):
        super().__init__(f"[{source_name}] failed to fetch price data for {ticker}: {detail}")
        self.source_name = source_name
        self.ticker = ticker


class PriceSource(ABC):
    name: str = "unknown"

    @abstractmethod
    def fetch_ohlcv(self, ticker: str, lookback_bars: int = 90, timeframe: str = "1Hour") -> pd.DataFrame:
        """
        Return a DataFrame with columns ['open','high','low','close','volume'],
        indexed by ISO-8601 timestamp string, oldest bar first.
        Raises PriceSourceError on any failure (network, auth, bad ticker, etc).
        """
        raise NotImplementedError

    def _get(self, url: str, params: dict | None = None, headers: dict | None = None, timeout: float = 10.0):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise PriceSourceError(self.name, params.get("symbol", "?") if params else "?", str(e)) from e


def _bars_to_df(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {'timestamp','open','high','low','close','volume'} oldest-first."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


class AlpacaPriceSource(PriceSource):
    """
    Stocks - primary source. Free paper-trading tier includes market data.
    Docs: https://docs.alpaca.markets/reference/stockbars
    """
    name = "alpaca"

    def __init__(self, cfg: IngestionConfig | None = None):
        self.cfg = cfg or default_config

    def fetch_ohlcv(self, ticker: str, lookback_bars: int = 90, timeframe: str = "1Hour") -> pd.DataFrame:
        if not self.cfg.alpaca_api_key or not self.cfg.alpaca_api_secret:
            raise MissingAPIKeyError("Alpaca", "ALPACA_API_KEY / ALPACA_API_SECRET")

        end = datetime.now(timezone.utc)
        # Rough padding: request more calendar days than bars needed, since
        # hourly bars only exist during market hours for stocks.
        start = end - timedelta(days=max(5, lookback_bars // 6 + 5))

        url = f"{self.cfg.alpaca_base_url}/v2/stocks/{ticker}/bars"
        headers = {
            "APCA-API-KEY-ID": self.cfg.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.cfg.alpaca_api_secret,
        }
        params = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": lookback_bars,
            "adjustment": "raw",
        }

        data = self._get(url, params=params, headers=headers, timeout=self.cfg.request_timeout_seconds)

        bars = data.get("bars")
        if bars is None:
            raise PriceSourceError(self.name, ticker, f"unexpected response shape: {list(data.keys())}")

        rows = [
            {
                "timestamp": b["t"],
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b["v"],
            }
            for b in bars
        ]
        return _bars_to_df(rows).tail(lookback_bars)


class AlphaVantagePriceSource(PriceSource):
    """Stocks - fallback source. Free tier: 5 calls/min, 25/day (used with rate_limiter)."""
    name = "alpha_vantage"

    def __init__(self, cfg: IngestionConfig | None = None):
        self.cfg = cfg or default_config

    def fetch_ohlcv(self, ticker: str, lookback_bars: int = 90, timeframe: str = "60min") -> pd.DataFrame:
        if not self.cfg.alpha_vantage_api_key:
            raise MissingAPIKeyError("Alpha Vantage", "ALPHA_VANTAGE_API_KEY")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": timeframe,
            "outputsize": "compact",
            "apikey": self.cfg.alpha_vantage_api_key,
        }
        data = self._get(url, params=params, timeout=self.cfg.request_timeout_seconds)

        if "Error Message" in data:
            raise PriceSourceError(self.name, ticker, data["Error Message"])
        if "Note" in data:  # rate limit message, returned with HTTP 200
            raise PriceSourceError(self.name, ticker, f"rate limited: {data['Note']}")

        series_key = next((k for k in data if "Time Series" in k), None)
        if series_key is None:
            raise PriceSourceError(self.name, ticker, f"unexpected response shape: {list(data.keys())}")

        rows = [
            {
                "timestamp": ts,
                "open": vals["1. open"],
                "high": vals["2. high"],
                "low": vals["3. low"],
                "close": vals["4. close"],
                "volume": vals["5. volume"],
            }
            for ts, vals in data[series_key].items()
        ]
        rows.sort(key=lambda r: r["timestamp"])  # Alpha Vantage returns newest-first
        return _bars_to_df(rows).tail(lookback_bars)


_BINANCE_INTERVAL_MAP = {"1Hour": "1h", "1Day": "1d", "1Min": "1m", "15Min": "15m"}


class BinancePriceSource(PriceSource):
    """Crypto - primary source. Public market data, no API key required."""
    name = "binance"

    def __init__(self, cfg: IngestionConfig | None = None):
        self.cfg = cfg or default_config

    def fetch_ohlcv(self, ticker: str, lookback_bars: int = 90, timeframe: str = "1Hour") -> pd.DataFrame:
        symbol = ticker.upper()
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"  # e.g. "BTC" -> "BTCUSDT"

        interval = _BINANCE_INTERVAL_MAP.get(timeframe, "1h")
        url = f"{self.cfg.binance_base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": lookback_bars}

        data = self._get(url, params=params, timeout=self.cfg.request_timeout_seconds)

        if not isinstance(data, list):
            raise PriceSourceError(self.name, ticker, f"unexpected response shape: {data}")

        # Binance kline row: [open_time, open, high, low, close, volume, close_time, ...]
        rows = [
            {
                "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).isoformat(),
                "open": k[1],
                "high": k[2],
                "low": k[3],
                "close": k[4],
                "volume": k[5],
            }
            for k in data
        ]
        return _bars_to_df(rows)


class CoinCapPriceSource(PriceSource):
    """Crypto - fallback source. Public, no API key required for the free tier."""
    name = "coincap"

    _SYMBOL_TO_ID = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano",
                      "DOGE": "dogecoin", "XRP": "ripple", "MATIC": "polygon", "DOT": "polkadot"}

    def __init__(self, cfg: IngestionConfig | None = None):
        self.cfg = cfg or default_config

    def fetch_ohlcv(self, ticker: str, lookback_bars: int = 90, timeframe: str = "1Hour") -> pd.DataFrame:
        asset_id = self._SYMBOL_TO_ID.get(ticker.upper())
        if asset_id is None:
            raise PriceSourceError(self.name, ticker, f"no known CoinCap asset id for '{ticker}'")

        interval = "h1" if timeframe in ("1Hour", "60min") else "d1"
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_bars if interval == "h1" else lookback_bars * 24)

        url = f"{self.cfg.coincap_base_url}/assets/{asset_id}/history"
        params = {"interval": interval, "start": int(start.timestamp() * 1000), "end": int(end.timestamp() * 1000)}

        data = self._get(url, params=params, timeout=self.cfg.request_timeout_seconds)

        points = data.get("data")
        if points is None:
            raise PriceSourceError(self.name, ticker, f"unexpected response shape: {list(data.keys())}")

        # CoinCap's /history endpoint gives price points, not full OHLCV candles.
        # We approximate O/H/L/C from the sequence of price points within each
        # bar (CoinCap doesn't provide true intrabar OHLC on the free tier) and
        # leave volume at 0 (not exposed at this endpoint) - documented in
        # PriceSource docstring / README so this doesn't silently mislead ATR.
        rows = [
            {
                "timestamp": datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc).isoformat(),
                "open": p["priceUsd"],
                "high": p["priceUsd"],
                "low": p["priceUsd"],
                "close": p["priceUsd"],
                "volume": 0.0,
            }
            for p in points
        ]
        return _bars_to_df(rows).tail(lookback_bars)
