"""
omnimarket.api
FastAPI layer for the prediction/risk/paper-trading engine.

Paper-trading simulation only. Portfolio and risk state are persisted in a
local SQLite database so a process restart no longer wipes the session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from models import (
    AssetClass, Direction, NewsItem, PortfolioState, Position, RiskEvent,
    Signal, TechnicalSnapshot, Trade,
)
from indicators import compute_snapshot
from sentiment import SentimentAggregator
from prediction_engine import PredictionEngine
from risk_manager import RiskManager, RiskLimits
from trade_engine import PaperTradingEngine
from learning_engine import AdaptiveLearningEngine
from ingestion.pipeline import IngestionPipeline
from persistence import StateStore

import pandas as pd

app = FastAPI(
    title="OmniMarket AI - Prediction & Trade Signal Engine",
    description="Backend for the OmniMarket AI paper-trading dashboard. Paper-trading simulation only. Not financial advice.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = StateStore()


def _enum(value, enum_type):
    return value if isinstance(value, enum_type) else enum_type(value)


def _restore_state() -> tuple[PortfolioState, RiskManager]:
    saved = store.get("portfolio")
    if not saved:
        return PortfolioState(cash_usd=10_000.0, starting_equity=10_000.0), RiskManager(limits=RiskLimits())

    portfolio = PortfolioState(
        cash_usd=float(saved["cash_usd"]),
        starting_equity=float(saved["starting_equity"]),
        positions=[
            Position(
                ticker=p["ticker"], direction=_enum(p["direction"], Direction),
                entry_price=float(p["entry_price"]), size_usd=float(p["size_usd"]),
                stop_loss=float(p["stop_loss"]), take_profit=float(p["take_profit"]),
                opened_at=p["opened_at"],
            ) for p in saved.get("positions", [])
        ],
        trade_log=[
            Trade(
                trade_id=t["trade_id"], ticker=t["ticker"],
                asset_class=_enum(t["asset_class"], AssetClass), direction=_enum(t["direction"], Direction),
                entry_price=float(t["entry_price"]), size_usd=float(t["size_usd"]),
                stop_loss=float(t["stop_loss"]), take_profit=float(t["take_profit"]),
                rationale=t["rationale"], confidence=float(t["confidence"]), opened_at=t["opened_at"],
                signal_components=t.get("signal_components", {}), status=t.get("status", "open"),
                exit_price=t.get("exit_price"), closed_at=t.get("closed_at"),
                pnl_usd=t.get("pnl_usd"), roi_pct=t.get("roi_pct"),
            ) for t in saved.get("trade_log", [])
        ],
    )
    risk = RiskManager(limits=RiskLimits())
    risk.active_risk_events = [RiskEvent(**e) for e in store.get("risk_events") or []]
    risk.decision_log = store.get("risk_decisions") or []
    return portfolio, risk


def _save_state() -> None:
    store.set("portfolio", {
        "cash_usd": portfolio.cash_usd,
        "starting_equity": portfolio.starting_equity,
        "positions": [p.__dict__ | {"direction": p.direction.value} for p in portfolio.positions],
        "trade_log": [t.__dict__ | {"direction": t.direction.value, "asset_class": t.asset_class.value} for t in portfolio.trade_log],
    })
    store.set("risk_events", [e.__dict__ for e in risk_manager.active_risk_events])
    store.set("risk_decisions", risk_manager.decision_log)


portfolio, risk_manager = _restore_state()
prediction_engine = PredictionEngine()
sentiment_aggregator = SentimentAggregator()
learning_engine = AdaptiveLearningEngine()
_persisted_weights = learning_engine.load_persisted_weights()
if _persisted_weights is not None:
    prediction_engine.set_weights(_persisted_weights)
trader = PaperTradingEngine(portfolio, risk_manager, learning_engine=learning_engine)
ingestion_pipeline = IngestionPipeline()
_last_tech_snapshots: dict[str, TechnicalSnapshot] = {}
_last_signals: dict[str, Signal] = {}


class OHLCVBarIn(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class NewsItemIn(BaseModel):
    ticker: str
    headline: str
    source: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    published_at: str
    event_type: str = "general"


class SignalsRequest(BaseModel):
    ohlcv: dict[str, list[OHLCVBarIn]]
    news: list[NewsItemIn] = []
    asset_classes: dict[str, AssetClass] = {}


class TradeRequest(BaseModel):
    ticker: str
    requested_size_usd: Optional[float] = Field(None, gt=0)


class IngestRequest(BaseModel):
    tickers: dict[str, AssetClass]
    lookback_bars: int = Field(90, ge=2, le=1000)
    news_lookback_hours: float = Field(48.0, gt=0, le=720)


class RiskEventIn(BaseModel):
    ticker: Optional[str] = None
    reason: str
    severity: str = "high"


class CloseTradeRequest(BaseModel):
    trade_id: str
    exit_price: float = Field(..., gt=0)


class MarkToMarketRequest(BaseModel):
    prices: dict[str, float]


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(), "persistence": "sqlite"}


@app.post("/signals")
def generate_signals(req: SignalsRequest):
    global _last_tech_snapshots, _last_signals
    if not req.ohlcv:
        raise HTTPException(400, "No OHLCV data supplied")
    tech_snapshots: dict[str, TechnicalSnapshot] = {}
    for ticker, bars in req.ohlcv.items():
        if len(bars) < 2:
            continue
        df = pd.DataFrame(
            [{"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume} for b in bars],
            index=[b.timestamp for b in bars],
        )
        try:
            tech_snapshots[ticker] = compute_snapshot(ticker, df)
        except ValueError:
            continue
    if not tech_snapshots:
        raise HTTPException(400, "Not enough OHLCV history to compute indicators (need >= 2 bars)")
    news_items = [NewsItem(**n.model_dump()) for n in req.news]
    sentiment_snapshots = sentiment_aggregator.aggregate(news_items)
    asset_classes = req.asset_classes or {t: AssetClass.STOCK for t in tech_snapshots}
    signals = prediction_engine.generate_batch(tech_snapshots, sentiment_snapshots, asset_classes)
    _last_tech_snapshots = tech_snapshots
    _last_signals = {s.ticker: s for s in signals}
    return {"signals": [s.__dict__ | {"direction": s.direction.value, "asset_class": s.asset_class.value, "regime": s.regime.value} for s in signals]}


@app.post("/ingest")
def ingest_and_generate_signals(req: IngestRequest):
    global _last_tech_snapshots, _last_signals
    results = ingestion_pipeline.ingest_batch(req.tickers, req.lookback_bars, req.news_lookback_hours)
    tech_snapshots, sentiment_snapshots, diagnostics = {}, {}, {}
    for ticker, result in results.items():
        diagnostics[ticker] = {"ok": result.ok, "price_source_used": result.price_source_used, "news_source_used": result.news_source_used, "news_articles_found": len(result.news), "errors": result.errors}
        if not result.ok:
            continue
        try:
            tech_snapshots[ticker] = compute_snapshot(ticker, result.ohlcv)
        except ValueError as e:
            diagnostics[ticker]["errors"].append(f"indicator computation failed: {e}")
            continue
        if result.news:
            sentiment_snapshots.update(sentiment_aggregator.aggregate(result.news))
    if not tech_snapshots:
        raise HTTPException(502, detail={"message": "No usable price data for any requested ticker.", "diagnostics": diagnostics})
    signals = prediction_engine.generate_batch(tech_snapshots, sentiment_snapshots, req.tickers)
    _last_tech_snapshots = tech_snapshots
    _last_signals = {s.ticker: s for s in signals}
    return {"signals": [s.__dict__ | {"direction": s.direction.value, "asset_class": s.asset_class.value, "regime": s.regime.value} for s in signals], "diagnostics": diagnostics}


@app.post("/risk-events")
def flag_risk_event(event: RiskEventIn):
    risk_manager.flag_risk_event(RiskEvent(**event.model_dump()))
    _save_state()
    return {"active_risk_events": len(risk_manager.active_risk_events)}


@app.delete("/risk-events/{ticker}")
def clear_risk_event(ticker: str):
    risk_manager.clear_risk_events(ticker)
    _save_state()
    return {"active_risk_events": len(risk_manager.active_risk_events)}


@app.post("/trades")
def execute_trade(req: TradeRequest):
    if req.ticker not in _last_tech_snapshots or req.ticker not in _last_signals:
        raise HTTPException(400, f"No recent signal for {req.ticker}. Call /signals first.")
    tech, signal = _last_tech_snapshots[req.ticker], _last_signals[req.ticker]
    trade, decision = trader.open_trade_from_signal(signal, tech, req.requested_size_usd)
    _save_state()
    return {"executed": trade is not None, "reason": decision.reason,
            "signal_used": signal.__dict__ | {"direction": signal.direction.value, "asset_class": signal.asset_class.value, "regime": signal.regime.value},
            "trade": trade.__dict__ | {"direction": trade.direction.value, "asset_class": trade.asset_class.value} if trade else None}


@app.post("/trades/close")
def close_trade(req: CloseTradeRequest):
    trade = trader.close_trade_manual(req.trade_id, req.exit_price)
    if not trade:
        raise HTTPException(404, f"No open trade found with id {req.trade_id}")
    _save_state()
    return {"trade": trade.__dict__ | {"direction": trade.direction.value, "asset_class": trade.asset_class.value}}


@app.post("/mark-to-market")
def mark_to_market(req: MarkToMarketRequest):
    if any(price <= 0 for price in req.prices.values()):
        raise HTTPException(400, "All mark-to-market prices must be positive")
    closed = trader.mark_to_market_and_check_exits(req.prices)
    _save_state()
    return {"closed_trades": [t.__dict__ | {"direction": t.direction.value, "asset_class": t.asset_class.value} for t in closed]}


@app.get("/portfolio")
def get_portfolio():
    return {"cash_usd": round(portfolio.cash_usd, 2), "equity": round(portfolio.equity, 2), "starting_equity": portfolio.starting_equity,
            "total_return_pct": round(portfolio.total_return_pct, 2), "exposure_pct": round(portfolio.exposure_pct, 2),
            "open_positions": len(portfolio.positions),
            "trade_log": [t.__dict__ | {"direction": t.direction.value, "asset_class": t.asset_class.value} for t in portfolio.trade_log]}


@app.get("/risk-log")
def get_risk_log():
    return {"decisions": risk_manager.decision_log, "active_events": [e.__dict__ for e in risk_manager.active_risk_events]}


@app.get("/learning/status")
def learning_status():
    n, wins, losses = learning_engine.sample_counts()
    weights = prediction_engine.get_weights()
    return {"closed_trades_logged": n, "wins": wins, "losses": losses, "min_samples_to_retrain": learning_engine.min_samples_to_retrain,
            "ready_to_retrain": n >= learning_engine.min_samples_to_retrain,
            "current_weights": {"sentiment": weights.sentiment, "technical": weights.technical, "regime": weights.regime},
            "training_history": [r.__dict__ for r in learning_engine.training_log]}


@app.post("/learning/retrain")
def trigger_retrain(force: bool = False):
    try:
        result = learning_engine.retrain(prediction_engine=prediction_engine) if force else learning_engine.maybe_retrain(prediction_engine=prediction_engine)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    if result is None:
        n, wins, losses = learning_engine.sample_counts()
        raise HTTPException(409, f"Not enough labeled data yet: {n} closed trades logged (need >= {learning_engine.min_samples_to_retrain}, with both classes represented). Use ?force=true to override.")
    return {"result": result.__dict__}
