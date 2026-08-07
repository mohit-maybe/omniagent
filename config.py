"""
omnimarket.ingestion.config
------------------------------
Loads API keys and ingestion settings from environment variables (or a
.env file via python-dotenv). Nothing here talks to the network - it's
pure configuration, so it's safe to import and test without any keys set.

Missing keys are NOT an error at import time - individual data sources
check for their own key and raise a clear, specific error only when
something actually tries to use them. That way `from ingestion import
pipeline` doesn't blow up just because you haven't set up Marketaux yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op if there's no .env file present; real env vars still apply


@dataclass(frozen=True)
class IngestionConfig:
    # Stocks
    alpaca_api_key: str | None = os.getenv("ALPACA_API_KEY")
    alpaca_api_secret: str | None = os.getenv("ALPACA_API_SECRET")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")
    alpha_vantage_api_key: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")

    # Crypto (Binance public market data needs no key; CoinCap free tier needs no key either)
    binance_base_url: str = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    coincap_base_url: str = os.getenv("COINCAP_BASE_URL", "https://api.coincap.io/v2")

    # News
    marketaux_api_key: str | None = os.getenv("MARKETAUX_API_KEY")
    finnhub_api_key: str | None = os.getenv("FINNHUB_API_KEY")

    # Rate limits (free-tier defaults from each provider's published docs -
    # override via env if your plan differs)
    marketaux_daily_limit: int = int(os.getenv("MARKETAUX_DAILY_LIMIT", "100"))
    finnhub_per_minute_limit: int = int(os.getenv("FINNHUB_PER_MINUTE_LIMIT", "60"))
    alpha_vantage_per_minute_limit: int = int(os.getenv("ALPHA_VANTAGE_PER_MINUTE_LIMIT", "5"))

    request_timeout_seconds: float = float(os.getenv("INGESTION_TIMEOUT_SECONDS", "10"))


config = IngestionConfig()


class MissingAPIKeyError(RuntimeError):
    """Raised when a source that requires a key is used without one configured."""
    def __init__(self, source_name: str, env_var: str):
        super().__init__(
            f"{source_name} requires an API key. Set the {env_var} environment "
            f"variable (or add it to a .env file) before using this source."
        )
