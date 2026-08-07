"""
omnimarket.demo
------------------
Runs the full pipeline (indicators -> sentiment -> prediction -> risk ->
paper trades) against SYNTHETIC data, so the engine can be verified with
zero API keys. Swap `generate_synthetic_ohlcv` / `generate_synthetic_news`
for real Alpaca/Binance/Marketaux/Finnhub pulls later - nothing else in
the pipeline needs to change.

Run: python demo.py
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from models import AssetClass, NewsItem, PortfolioState, RiskEvent
from indicators import compute_snapshot
from sentiment import SentimentAggregator
from prediction_engine import PredictionEngine
from risk_manager import RiskManager, RiskLimits
from trade_engine import PaperTradingEngine

random.seed(7)
np.random.seed(7)

TICKERS = {
    "AAPL": AssetClass.STOCK,
    "TSLA": AssetClass.STOCK,
    "NVDA": AssetClass.STOCK,
    "SPY": AssetClass.STOCK,
    "BTC": AssetClass.CRYPTO,
    "ETH": AssetClass.CRYPTO,
}

BASE_PRICES = {"AAPL": 210.0, "TSLA": 250.0, "NVDA": 130.0, "SPY": 560.0, "BTC": 64000.0, "ETH": 3400.0}


# Tickers engineered with a strong, sustained uptrend so the demo can prove
# out actual trade execution (AAPL) and risk-event blocking of an otherwise
# qualifying signal (TSLA), rather than showing only HOLDs from noise.
SCENARIO_DRIFT = {"AAPL": 0.006, "TSLA": 0.006}


def generate_synthetic_ohlcv(ticker: str, bars: int = 90, drift: float = 0.0004, vol: float = 0.018) -> pd.DataFrame:
    base = BASE_PRICES[ticker]
    drift = SCENARIO_DRIFT.get(ticker, drift)
    now = datetime.now(timezone.utc)
    returns = np.random.normal(drift, vol, bars)
    closes = base * np.cumprod(1 + returns)
    highs = closes * (1 + np.abs(np.random.normal(0, vol / 2, bars)))
    lows = closes * (1 - np.abs(np.random.normal(0, vol / 2, bars)))
    opens = np.roll(closes, 1)
    opens[0] = base
    volumes = np.random.uniform(1e6, 5e6, bars)
    timestamps = [(now - timedelta(hours=(bars - i))).isoformat() for i in range(bars)]

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=timestamps,
    )


HEADLINE_BANK = {
    "bullish": [
        "beats earnings expectations, raises guidance",
        "announces major product breakthrough",
        "gets analyst upgrade on strong demand outlook",
    ],
    "bearish": [
        "misses revenue targets amid weak demand",
        "faces regulatory scrutiny over practices",
        "downgraded on margin compression concerns",
    ],
    "neutral": [
        "holds investor day, reiterates prior guidance",
        "trades sideways ahead of Fed decision",
    ],
}


def generate_synthetic_news(ticker: str, n: int = 6) -> list[NewsItem]:
    now = datetime.now(timezone.utc)
    items = []
    for i in range(n):
        if ticker in SCENARIO_DRIFT:
            tone = "bullish"  # scenario tickers: consistent, recent bullish coverage
        else:
            tone = random.choices(["bullish", "bearish", "neutral"], weights=[0.4, 0.3, 0.3])[0]
        score = {
            "bullish": random.uniform(0.3, 0.9),
            "bearish": random.uniform(-0.9, -0.3),
            "neutral": random.uniform(-0.15, 0.15),
        }[tone]
        headline = f"{ticker} {random.choice(HEADLINE_BANK[tone])}"
        recency_hours = random.uniform(0, 12) if ticker in SCENARIO_DRIFT else random.uniform(0, 40)
        published = (now - timedelta(hours=recency_hours)).isoformat()
        event_type = random.choice(["earnings", "regulation", "macro", "general"])
        items.append(
            NewsItem(
                ticker=ticker,
                headline=headline,
                source=random.choice(["Marketaux", "APITube", "Finnhub"]),
                sentiment_score=round(score, 3),
                published_at=published,
                event_type=event_type,
            )
        )
    return items


def run_daily_briefing() -> None:
    print("=" * 72)
    print("OmniMarket AI - Daily Briefing (synthetic data demo)")
    print("=" * 72)

    # 1. Build technical snapshots per ticker
    tech_snapshots = {}
    ohlcv_by_ticker = {}
    for ticker in TICKERS:
        df = generate_synthetic_ohlcv(ticker)
        ohlcv_by_ticker[ticker] = df
        tech_snapshots[ticker] = compute_snapshot(ticker, df)

    # 2. Build sentiment snapshots per ticker
    all_news = []
    for ticker in TICKERS:
        all_news.extend(generate_synthetic_news(ticker))
    aggregator = SentimentAggregator(lookback_hours=24, prior_window_hours=24)
    sentiment_snapshots = aggregator.aggregate(all_news)

    # 3. Generate signals
    engine = PredictionEngine()
    signals = engine.generate_batch(tech_snapshots, sentiment_snapshots, TICKERS)

    # 4. Risk manager + one synthetic high-risk event to prove blocking works
    # NOTE: min_confidence_to_trade is lowered here ONLY so this synthetic demo
    # exercises the full open->stop/target->close path. Production default is 55
    # (see RiskLimits in risk_manager.py) - do not ship this lower threshold.
    DEMO_MIN_CONFIDENCE = 12
    risk_manager = RiskManager(limits=RiskLimits(max_position_pct=10, max_total_exposure_pct=50,
                                                   min_confidence_to_trade=DEMO_MIN_CONFIDENCE))
    risk_manager.flag_risk_event(RiskEvent(ticker="TSLA", reason="Earnings gap risk - report due after close",
                                            severity="high"))

    # 5. Portfolio + paper trading engine ($10,000 virtual, matching spec)
    portfolio = PortfolioState(cash_usd=10_000.0, starting_equity=10_000.0)
    trader = PaperTradingEngine(portfolio, risk_manager)

    print("\n-- Signals (sorted by confidence) --")
    for s in signals:
        print(f"[{s.confidence:5.1f}%] {s.direction.value:4s} {s.ticker:5s} "
              f"({s.regime.value:8s}) - {s.rationale}")

    print("\n-- Trade Execution --")
    for s in signals:
        trade, decision = trader.open_trade_from_signal(s, tech_snapshots[s.ticker])
        status = "EXECUTED" if trade else "SKIPPED "
        print(f"{status} {s.ticker:5s} {s.direction.value:4s} -> {decision.reason}")

    # 6. Simulate price movement one step forward and mark-to-market
    next_prices = {t: tech_snapshots[t].last_price * (1 + random.uniform(-0.03, 0.03)) for t in TICKERS}
    closed = trader.mark_to_market_and_check_exits(next_prices)

    print("\n-- Portfolio Snapshot --")
    print(f"Equity: ${portfolio.equity:,.2f} | Total Return: {portfolio.total_return_pct:+.2f}% | "
          f"Exposure: {portfolio.exposure_pct:.1f}% | Cash: ${portfolio.cash_usd:,.2f}")
    print(f"Open positions: {len(portfolio.positions)} | Closed this cycle: {len(closed)}")

    open_trades = [t for t in portfolio.trade_log if t.status == "open"]
    if open_trades:
        print("\nOpen trades:")
        for t in open_trades:
            print(f"  {t.trade_id} {t.ticker:5s} {t.direction.value:4s} entry=${t.entry_price:,.2f} "
                  f"size=${t.size_usd:,.0f} SL=${t.stop_loss:,.2f} TP=${t.take_profit:,.2f}")

    if closed:
        print("\nClosed this cycle:")
        for t in closed:
            print(f"  {t.trade_id} {t.ticker:5s} {t.status:12s} PnL=${t.pnl_usd:+,.2f} ROI={t.roi_pct:+.2f}%")

    print("\nRisk decision log:")
    for entry in risk_manager.decision_log:
        print(f"  {entry}")

    # -- Deterministic proof that an active high-risk event blocks a trade even
    # when confidence would otherwise clear the threshold (TSLA earnings gap) --
    from models import AssetClass as _AC, Direction as _Dir, MarketRegime as _MR, Signal as _Signal
    forced_signal = _Signal(
        ticker="TSLA", asset_class=_AC.STOCK, direction=_Dir.BUY,
        confidence=90.0, regime=_MR.BULL,
        rationale="[demo] Hypothetical high-confidence signal to prove risk-event blocking.",
    )
    forced_trade, forced_decision = trader.open_trade_from_signal(forced_signal, tech_snapshots["TSLA"])
    print("\n-- Risk-event block proof (forced 90% confidence TSLA BUY) --")
    print(f"{'EXECUTED' if forced_trade else 'BLOCKED '} -> {forced_decision.reason}")

    print("\n" + "=" * 72)
    print("DISCLAIMER: Paper-trading simulation only. Not financial advice.")
    print("Past performance does not guarantee future results.")
    print("=" * 72)


def run_learning_demo(cycles: int = 60) -> None:
    """
    Proves the closed-loop learning system: runs `cycles` trades through the
    REAL open/close plumbing (not a shortcut), lets outcomes be biased so
    that the technical component is actually predictive and sentiment is
    noise, retrains, and shows the weight shift. This is what "learns from
    its mistakes" means concretely - the engine's own component weights
    change based on which inputs its trade history shows were reliable.
    """
    import random as _random
    from models import AssetClass as _AC, Direction as _Dir, MarketRegime as _MR, PortfolioState as _PS, \
        Signal as _Signal, TechnicalSnapshot as _TS
    from risk_manager import RiskManager as _RM, RiskLimits as _RL
    from trade_engine import PaperTradingEngine as _PTE
    from prediction_engine import PredictionEngine as _PE, EngineWeights as _EW
    from learning_engine import AdaptiveLearningEngine

    print("\n" + "=" * 72)
    print(f"LEARNING DEMO - simulating {cycles} closed trades, then retraining")
    print("=" * 72)

    # Isolated store so this demo run doesn't collide with a real deployment's
    # learning history if this file is imported elsewhere.
    demo_store = "demo_learning_store.json"
    demo_weights = "demo_learned_weights.json"
    for f in (demo_store, demo_weights, "demo_training_log.json"):
        if __import__("os").path.exists(f):
            __import__("os").remove(f)

    learning = AdaptiveLearningEngine(store_path=demo_store, weights_path=demo_weights,
                                       min_samples_to_retrain=20)
    portfolio = _PS(cash_usd=200_000.0, starting_equity=200_000.0)  # large so sizing never blocks the demo
    risk = _RM(_RL(max_position_pct=100, max_total_exposure_pct=10_000, min_confidence_to_trade=0))
    trader = _PTE(portfolio, risk, learning_engine=learning, default_position_pct=0.1)
    engine = _PE()

    print(f"Starting weights: {engine.get_weights()}")

    for i in range(cycles):
        # Hidden ground truth for this demo: technical score genuinely predicts
        # the outcome; sentiment is noise; regime has a small real effect.
        s_tech = _random.uniform(-1, 1)
        s_sent = _random.uniform(-1, 1)
        s_regime = _random.uniform(-1, 1)
        win_prob = max(0.05, min(0.95, 0.5 + 0.35 * s_tech + 0.08 * s_regime))
        will_win = _random.random() < win_prob

        tech = _TS(ticker="SIM", rsi=50, macd=0.1, macd_signal=0.0, sma_fast=101, sma_slow=100,
                   atr=1.0, last_price=100.0)
        signal = _Signal(
            ticker="SIM", asset_class=_AC.STOCK, direction=_Dir.BUY, confidence=70.0,
            regime=_MR.BULL, rationale="[learning-demo]",
            components={"sentiment": s_sent, "technical": s_tech, "regime": s_regime},
        )
        trade, _decision = trader.open_trade_from_signal(signal, tech, requested_size_usd=100.0)
        if not trade:
            continue
        exit_price = tech.last_price * (1.05 if will_win else 0.95)
        trader.close_trade_manual(trade.trade_id, exit_price)

    n, wins, losses = learning.sample_counts()
    print(f"\nClosed {n} trades ({wins} wins / {losses} losses) - retraining...")

    result = learning.maybe_retrain(prediction_engine=engine)
    if result is None:
        print("Not enough balanced data to retrain yet.")
        return

    print(f"\nTrain accuracy on logged outcomes: {result.train_accuracy:.1%}")
    print(f"Learned coefficients (higher |value| = more historically predictive): {result.coefficients}")
    print(f"\nWeights BEFORE learning: sentiment=0.40 technical=0.45 regime=0.15 (engine defaults)")
    print(f"Weights AFTER learning:  sentiment={result.weights['sentiment']:.2f} "
          f"technical={result.weights['technical']:.2f} regime={result.weights['regime']:.2f}")
    print(f"\nPredictionEngine.weights is now live-updated: {engine.get_weights()}")
    print("\nPersisted to demo_learned_weights.json - in production, load this at")
    print("startup via AdaptiveLearningEngine().load_persisted_weights() and call")
    print("prediction_engine.set_weights(...) so learning survives a restart.")


if __name__ == "__main__":
    run_daily_briefing()
    run_learning_demo()
