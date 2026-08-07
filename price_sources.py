"""Deprecated compatibility shim.

Use `ingestion.price_sources` for the canonical ingestion price sources.
"""

from __future__ import annotations

from ingestion.price_sources import *

__all__ = [
    "PriceSourceError",
    "PriceSource",
    "AlpacaPriceSource",
    "AlphaVantagePriceSource",
    "BinancePriceSource",
    "CoinCapPriceSource",
]
