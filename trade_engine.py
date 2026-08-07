"""
omnimarket.trade_engine
--------------------------
Simulates trades based on approved signals. Never touches real money or
real brokerage APIs - this is the "paper-trading" layer.

Responsibilities:
  - Size positions (capped by RiskManager)
  - Set stop-loss / take-profit levels via ATR multiples
  - Open/close positions against a virtual PortfolioState
  - Log every trade with rationale, so the dashboard's Trade Journal
    can show *why* each trade happened, not just the P&L
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

from models import (
    AssetClass,
    Direction,
    PortfolioState,
    Position,
    Signal,
    TechnicalSnapshot,
    Trade,
)
from risk_manager import RiskDecision, RiskManager


@dataclass
class ATRStopConfig:
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0  # ~2:1 reward/risk by default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperTradingEngine:
    def __init__(
        self,
        portfolio: PortfolioState,
        risk_manager: RiskManager,
        stop_config: ATRStopConfig | None = None,
        default_position_pct: float = 8.0,
        learning_engine=None,  # Optional["AdaptiveLearningEngine"] - avoids a hard import cycle
    ):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.stop_config = stop_config or ATRStopConfig()
        self.default_position_pct = default_position_pct
        self.learning_engine = learning_engine

    def _compute_stops(self, direction: Direction, entry_price: float, atr: float) -> tuple[float, float]:
        sl_dist = atr * self.stop_config.stop_loss_atr_mult
        tp_dist = atr * self.stop_config.take_profit_atr_mult
        if direction == Direction.BUY:
            return entry_price - sl_dist, entry_price + tp_dist
        else:  # SELL / short
            return entry_price + sl_dist, entry_price - tp_dist

    def open_trade_from_signal(
        self,
        signal: Signal,
        tech: TechnicalSnapshot,
        requested_size_usd: float | None = None,
    ) -> tuple[Trade | None, RiskDecision]:
        """
        Attempt to open a trade for a BUY/SELL signal. Returns (Trade, decision)
        - Trade is None if the RiskManager blocked it; the decision explains why.
        HOLD signals are never traded (returns (None, decision) with a HOLD reason).
        """
        if signal.direction == Direction.HOLD:
            decision = RiskDecision(approved=False, reason="HOLD signal - no trade taken")
            self.risk_manager._log(signal, decision)
            return None, decision

        size_request = requested_size_usd or (
            self.portfolio.starting_equity * (self.default_position_pct / 100)
        )
        decision = self.risk_manager.evaluate(signal, self.portfolio, size_request)
        if not decision.approved:
            return None, decision

        stop_loss, take_profit = self._compute_stops(signal.direction, tech.last_price, tech.atr)

        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            ticker=signal.ticker,
            asset_class=signal.asset_class,
            direction=signal.direction,
            entry_price=tech.last_price,
            size_usd=decision.approved_size_usd,
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            rationale=signal.rationale,
            confidence=signal.confidence,
            opened_at=_now(),
            signal_components=dict(signal.components),
            status="open",
        )

        position = Position(
            ticker=signal.ticker,
            direction=signal.direction,
            entry_price=trade.entry_price,
            size_usd=trade.size_usd,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            opened_at=trade.opened_at,
        )

        self.portfolio.cash_usd -= trade.size_usd
        self.portfolio.positions.append(position)
        self.portfolio.trade_log.append(trade)
        return trade, decision

    def mark_to_market_and_check_exits(self, current_prices: dict[str, float]) -> list[Trade]:
        """
        Given latest prices, close any open position that has hit its
        stop-loss or take-profit. Returns the list of trades closed this call.
        """
        closed: list[Trade] = []
        still_open_positions = []

        for position in self.portfolio.positions:
            price = current_prices.get(position.ticker)
            if price is None:
                still_open_positions.append(position)
                continue

            hit_stop = (
                price <= position.stop_loss if position.direction == Direction.BUY
                else price >= position.stop_loss
            )
            hit_target = (
                price >= position.take_profit if position.direction == Direction.BUY
                else price <= position.take_profit
            )

            if hit_stop or hit_target:
                trade = self._find_open_trade(position)
                if trade:
                    self._close_trade(trade, price, win=hit_target)
                    closed.append(trade)
            else:
                still_open_positions.append(position)

        self.portfolio.positions = still_open_positions
        return closed

    def close_trade_manual(self, trade_id: str, exit_price: float) -> Trade | None:
        trade = next((t for t in self.portfolio.trade_log if t.trade_id == trade_id and t.status == "open"), None)
        if not trade:
            return None
        win = (exit_price > trade.entry_price) == (trade.direction == Direction.BUY)
        self._close_trade(trade, exit_price, win=win, manual=True)
        self.portfolio.positions = [p for p in self.portfolio.positions if p.ticker != trade.ticker
                                     or p.opened_at != trade.opened_at]
        return trade

    def _find_open_trade(self, position: Position) -> Trade | None:
        for t in self.portfolio.trade_log:
            if t.ticker == position.ticker and t.opened_at == position.opened_at and t.status == "open":
                return t
        return None

    def _close_trade(self, trade: Trade, exit_price: float, win: bool, manual: bool = False) -> None:
        direction_mult = 1 if trade.direction == Direction.BUY else -1
        pct_move = (exit_price - trade.entry_price) / trade.entry_price * direction_mult
        pnl = trade.size_usd * pct_move

        trade.exit_price = round(exit_price, 4)
        trade.closed_at = _now()
        trade.pnl_usd = round(pnl, 2)
        trade.roi_pct = round(pct_move * 100, 2)
        trade.status = "closed_manual" if manual else ("closed_win" if win else "closed_loss")

        self.portfolio.cash_usd += trade.size_usd + pnl

        if self.learning_engine is not None:
            self.learning_engine.record_outcome(trade)
