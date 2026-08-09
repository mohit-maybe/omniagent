from models import AssetClass, Direction, Position, PortfolioState, RiskEvent, Trade
from persistence import StateStore


def test_state_store_round_trip(tmp_path):
    store = StateStore(str(tmp_path / "state.db"))
    portfolio = PortfolioState(cash_usd=9000.0, starting_equity=10000.0)
    portfolio.positions.append(Position("AAPL", Direction.BUY, 100.0, 1000.0, 95.0, 110.0))
    portfolio.trade_log.append(
        Trade("t1", "AAPL", AssetClass.STOCK, Direction.BUY, 100.0, 1000.0, 95.0, 110.0, "test", 80.0, "now")
    )
    store.set("portfolio", {
        "cash_usd": portfolio.cash_usd,
        "starting_equity": portfolio.starting_equity,
        "positions": [p.__dict__ | {"direction": p.direction.value} for p in portfolio.positions],
        "trade_log": [t.__dict__ | {"direction": t.direction.value, "asset_class": t.asset_class.value} for t in portfolio.trade_log],
    })
    store.set("risk_events", [RiskEvent("AAPL", "earnings").__dict__])

    restarted = StateStore(str(tmp_path / "state.db"))
    saved = restarted.get("portfolio")
    events = restarted.get("risk_events")

    assert saved["cash_usd"] == 9000.0
    assert saved["positions"][0]["ticker"] == "AAPL"
    assert saved["trade_log"][0]["trade_id"] == "t1"
    assert events[0]["reason"] == "earnings"
