from models import AssetClass, Direction, PortfolioState, RiskEvent, Signal, TechnicalSnapshot
from prediction_engine import PredictionEngine
from risk_manager import RiskLimits, RiskManager
from trade_engine import PaperTradingEngine


def make_signal(confidence=80.0, direction=Direction.BUY):
    return Signal(
        ticker="TEST",
        asset_class=AssetClass.STOCK,
        direction=direction,
        confidence=confidence,
        regime="bull",
        rationale="test",
        components={"sentiment": 0.2, "technical": 0.6, "regime": 0.3},
    )


def make_tech(price=100.0, atr_value=2.0):
    return TechnicalSnapshot(
        ticker="TEST",
        rsi=50.0,
        macd=1.0,
        macd_signal=0.5,
        sma_fast=101.0,
        sma_slow=99.0,
        atr=atr_value,
        last_price=price,
    )


def test_prediction_engine_returns_bounded_signal():
    signal = PredictionEngine().generate_signal("TEST", AssetClass.STOCK, make_tech(), None)
    assert 0.0 <= signal.confidence <= 100.0
    assert signal.direction in (Direction.BUY, Direction.SELL, Direction.HOLD)
    assert set(signal.components) == {"sentiment", "technical", "regime"}


def test_risk_manager_blocks_high_risk_event():
    portfolio = PortfolioState(cash_usd=10_000, starting_equity=10_000)
    risk = RiskManager(RiskLimits())
    risk.flag_risk_event(RiskEvent(ticker="TEST", reason="earnings"))

    decision = risk.evaluate(make_signal(), portfolio, 800)

    assert not decision.approved
    assert "high-risk event" in decision.reason


def test_risk_manager_caps_position_size():
    portfolio = PortfolioState(cash_usd=10_000, starting_equity=10_000)
    risk = RiskManager(RiskLimits(max_position_pct=10, max_total_exposure_pct=50))

    decision = risk.evaluate(make_signal(), portfolio, 2_000)

    assert decision.approved
    assert decision.approved_size_usd == 1_000


def test_paper_trade_opens_and_closes_at_target():
    portfolio = PortfolioState(cash_usd=10_000, starting_equity=10_000)
    risk = RiskManager()
    trader = PaperTradingEngine(portfolio, risk)

    trade, decision = trader.open_trade_from_signal(make_signal(), make_tech())

    assert decision.approved
    assert trade is not None
    assert len(portfolio.positions) == 1
    assert portfolio.cash_usd == 9_200

    closed = trader.mark_to_market_and_check_exits({"TEST": 106.0})

    assert len(closed) == 1
    assert closed[0].status == "closed_win"
    assert closed[0].pnl_usd == 48.0
    assert portfolio.cash_usd == 10_048
    assert not portfolio.positions


def test_paper_trade_stop_loss_for_short_position():
    portfolio = PortfolioState(cash_usd=10_000, starting_equity=10_000)
    risk = RiskManager()
    trader = PaperTradingEngine(portfolio, risk)

    trade, decision = trader.open_trade_from_signal(
        make_signal(direction=Direction.SELL), make_tech()
    )

    assert decision.approved
    assert trade is not None
    assert trade.stop_loss == 103.0
    assert trade.take_profit == 94.0

    closed = trader.mark_to_market_and_check_exits({"TEST": 103.0})

    assert len(closed) == 1
    assert closed[0].status == "closed_loss"
    assert closed[0].pnl_usd == -24.0
