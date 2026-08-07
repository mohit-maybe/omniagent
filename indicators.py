"""
omnimarket.indicators
----------------------
Pure technical-analysis functions operating on pandas DataFrames of OHLCV
data. No network calls, no state - easy to unit test and easy to swap the
upstream data source (Alpaca / Polygon / Binance / CoinCap) without
touching this file.

Expected DataFrame columns: ['open', 'high', 'low', 'close', 'volume'],
indexed by timestamp, ascending order (oldest first).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import TechnicalSnapshot


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)  # neutral until enough history


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_snapshot(ticker: str, df: pd.DataFrame,
                      sma_fast_window: int = 10,
                      sma_slow_window: int = 50) -> TechnicalSnapshot:
    """
    Reduce a full OHLCV history to a single TechnicalSnapshot representing
    the most recent bar. Requires enough history for the slow SMA window;
    falls back gracefully (repeats last valid value) on short series so the
    demo/dev path doesn't crash on thin data.
    """
    if len(df) < 2:
        raise ValueError(f"Not enough data to compute indicators for {ticker}")

    close = df["close"]
    rsi_series = rsi(close)
    macd_line, signal_line, _ = macd(close)
    sma_f = sma(close, min(sma_fast_window, len(close)))
    sma_s = sma(close, min(sma_slow_window, len(close)))
    atr_series = atr(df)

    def last_valid(s: pd.Series, fallback: float) -> float:
        s = s.dropna()
        return float(s.iloc[-1]) if len(s) else fallback

    last_price = float(close.iloc[-1])

    return TechnicalSnapshot(
        ticker=ticker,
        rsi=last_valid(rsi_series, 50.0),
        macd=last_valid(macd_line, 0.0),
        macd_signal=last_valid(signal_line, 0.0),
        sma_fast=last_valid(sma_f, last_price),
        sma_slow=last_valid(sma_s, last_price),
        atr=last_valid(atr_series, close.std() or last_price * 0.02),
        last_price=last_price,
    )
