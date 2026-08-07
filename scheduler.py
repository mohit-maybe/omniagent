"""
omnimarket.ingestion.scheduler
----------------------------------
Runs the ingestion -> signal -> risk -> trade pipeline on a timer, so
OmniMarket AI doesn't need a human to trigger each cycle. Two jobs:

  - daily_briefing_job: runs once/day (default 00:00 UTC, matching the
    spec) across the full watchlist - generates signals and attempts
    trades for each ticker.
  - mark_to_market_job: runs frequently (default every 5 min during
    market hours) to check open positions against current prices and
    close anything that's hit its stop/target.

This is a long-running process (`python scheduler.py`) separate from the
FastAPI server - in production you'd run api.py behind uvicorn/gunicorn
and this scheduler as a separate worker process (or a proper cron/Celery
beat job, per the original spec's "Celery for async news ingestion").
APScheduler is used here because it's a single dependency that's enough
for this job's needs without standing up Redis/RabbitMQ for a demo-scale
deployment - swap for Celery beat when you outgrow a single process.
"""

from __future__ import annotations

import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from models import AssetClass, PortfolioState
from indicators import compute_snapshot
from sentiment import SentimentAggregator
from prediction_engine import PredictionEngine
from risk_manager import RiskManager, RiskLimits
from trade_engine import PaperTradingEngine
from learning_engine import AdaptiveLearningEngine
from ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("omnimarket.scheduler")

# Default watchlist - adjust to taste, or load from a config file / DB later.
WATCHLIST: dict[str, AssetClass] = {
    "AAPL": AssetClass.STOCK, "TSLA": AssetClass.STOCK, "NVDA": AssetClass.STOCK, "SPY": AssetClass.STOCK,
    "BTC": AssetClass.CRYPTO, "ETH": AssetClass.CRYPTO,
}


class SchedulerState:
    """Holds the shared engine instances so both jobs operate on the same portfolio/weights."""

    def __init__(self):
        self.pipeline = IngestionPipeline()
        self.prediction_engine = PredictionEngine()
        self.risk_manager = RiskManager(limits=RiskLimits())
        self.portfolio = PortfolioState(cash_usd=10_000.0, starting_equity=10_000.0)
        self.learning_engine = AdaptiveLearningEngine()
        self.trader = PaperTradingEngine(self.portfolio, self.risk_manager, learning_engine=self.learning_engine)

        persisted = self.learning_engine.load_persisted_weights()
        if persisted is not None:
            self.prediction_engine.set_weights(persisted)
            logger.info("Loaded persisted learned weights: %s", persisted)

        self._last_tech_snapshots = {}


state = SchedulerState()


def daily_briefing_job() -> None:
    logger.info("Running daily briefing for %d tickers", len(WATCHLIST))
    results = state.pipeline.ingest_batch(WATCHLIST)

    tech_snapshots = {}
    sentiment_snapshots = {}
    for ticker, result in results.items():
        if not result.ok:
            logger.warning("Skipping %s - no price data (%s)", ticker, result.errors)
            continue
        try:
            tech_snapshots[ticker] = compute_snapshot(ticker, result.ohlcv)
        except ValueError as e:
            logger.warning("Skipping %s - indicator computation failed: %s", ticker, e)
            continue
        if result.news:
            snaps = SentimentAggregator().aggregate(result.news)
            sentiment_snapshots.update(snaps)

    if not tech_snapshots:
        logger.error("No tickers had usable data this cycle - skipping signal generation")
        return

    signals = state.prediction_engine.generate_batch(tech_snapshots, sentiment_snapshots, WATCHLIST)
    state._last_tech_snapshots = tech_snapshots

    logger.info("Top signals:")
    for s in signals[:5]:
        logger.info("  [%5.1f%%] %s %s (%s)", s.confidence, s.direction.value, s.ticker, s.regime.value)

    for s in signals:
        trade, decision = state.trader.open_trade_from_signal(s, tech_snapshots[s.ticker])
        if trade:
            logger.info("EXECUTED %s %s @ $%.2f (size $%.0f)", s.ticker, s.direction.value,
                        trade.entry_price, trade.size_usd)
        else:
            logger.info("SKIPPED %s: %s", s.ticker, decision.reason)

    retrain_result = state.learning_engine.maybe_retrain(state.prediction_engine)
    if retrain_result:
        logger.info("Learning engine retrained: accuracy=%.1f%% weights=%s",
                    retrain_result.train_accuracy * 100, retrain_result.weights)


def mark_to_market_job() -> None:
    if not state._last_tech_snapshots:
        return  # nothing ingested yet this run

    current_prices = {}
    for ticker in list(state._last_tech_snapshots.keys()):
        asset_class = WATCHLIST.get(ticker, AssetClass.STOCK)
        df, source, errors = state.pipeline.fetch_price(ticker, asset_class, lookback_bars=2)
        if df is not None and not df.empty:
            current_prices[ticker] = float(df.iloc[-1]["close"])
        else:
            logger.debug("Mark-to-market: no fresh price for %s (%s)", ticker, errors)

    if not current_prices:
        return

    closed = state.trader.mark_to_market_and_check_exits(current_prices)
    for t in closed:
        logger.info("CLOSED %s %s: %s PnL=$%.2f", t.ticker, t.status, t.trade_id, t.pnl_usd)


def run(daily_cron: str = "0 0 * * *", mark_to_market_minutes: int = 5) -> None:
    """
    daily_cron: standard 5-field cron expression, default midnight UTC
                (matches the spec's "daily briefing at 00:00 UTC").
    mark_to_market_minutes: how often to check open positions against
                current prices, in minutes.
    """
    scheduler = BlockingScheduler(timezone="UTC")

    minute, hour, day, month, day_of_week = daily_cron.split()
    scheduler.add_job(
        daily_briefing_job,
        trigger=CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week),
        id="daily_briefing",
        name="Daily signal generation + trade execution",
    )
    scheduler.add_job(
        mark_to_market_job,
        trigger=IntervalTrigger(minutes=mark_to_market_minutes),
        id="mark_to_market",
        name="Check open positions against current prices",
    )

    logger.info("Scheduler starting. Daily briefing: %s (UTC cron). Mark-to-market every %d min.",
                daily_cron, mark_to_market_minutes)
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    if "--once" in sys.argv:
        # Run a single cycle immediately and exit - useful for testing/cron
        # instead of APScheduler, or a one-off manual trigger.
        daily_briefing_job()
        mark_to_market_job()
    else:
        run()
