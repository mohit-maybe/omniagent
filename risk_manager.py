"""
omnimarket.risk_manager
-------------------------
Enforces the "Risk & Ethics Safeguards" from the spec, independently of
the prediction engine so a bad/over-confident signal can never bypass
portfolio-level limits:

  - Max 10% of starting equity per single asset
  - Max 50% total portfolio exposure
  - Trades blocked outright during flagged high-risk events
    (earnings gaps, black-swan/macro shocks, or anything the caller flags)
  - Every allow/block decision is logged for auditability

This module is deliberately conservative: when in doubt, it blocks the
trade rather than allowing it. It never generates or edits price/return
numbers - it only allows, shrinks, or blocks a proposed position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models import PortfolioState, RiskEvent, Signal


@dataclass
class RiskLimits:
    max_position_pct: float = 10.0   # % of starting equity per asset
    max_total_exposure_pct: float = 50.0  # % of starting equity across all open positions
    min_confidence_to_trade: float = 55.0  # HOLD/low-conviction signals never trade


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    approved_size_usd: float = 0.0


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.active_risk_events: list[RiskEvent] = []
        self.decision_log: list[dict] = []

    def flag_risk_event(self, event: RiskEvent) -> None:
        self.active_risk_events.append(event)

    def clear_risk_events(self, ticker: Optional[str] = None) -> None:
        if ticker is None:
            self.active_risk_events.clear()
        else:
            self.active_risk_events = [e for e in self.active_risk_events if e.ticker != ticker]

    def _blocking_event_for(self, ticker: str) -> Optional[RiskEvent]:
        for event in self.active_risk_events:
            if event.severity == "high" and (event.ticker is None or event.ticker == ticker):
                return event
        return None

    def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        requested_size_usd: float,
    ) -> RiskDecision:
        """
        Approve, shrink, or block a proposed trade for `signal.ticker`.
        Logs every decision (approved or not) for the audit trail.
        """
        decision: RiskDecision

        blocking_event = self._blocking_event_for(signal.ticker)
        if blocking_event:
            decision = RiskDecision(
                approved=False,
                reason=f"Blocked - active high-risk event: {blocking_event.reason}",
            )
            self._log(signal, decision)
            return decision

        if signal.confidence < self.limits.min_confidence_to_trade:
            decision = RiskDecision(
                approved=False,
                reason=f"Confidence {signal.confidence:.1f}% below threshold "
                       f"{self.limits.min_confidence_to_trade:.1f}%",
            )
            self._log(signal, decision)
            return decision

        max_position_usd = portfolio.starting_equity * (self.limits.max_position_pct / 100)
        current_exposure_usd = sum(p.size_usd for p in portfolio.positions)
        max_total_usd = portfolio.starting_equity * (self.limits.max_total_exposure_pct / 100)
        remaining_capacity_usd = max(0.0, max_total_usd - current_exposure_usd)

        approved_size = min(requested_size_usd, max_position_usd, remaining_capacity_usd, portfolio.cash_usd)

        if approved_size <= 0:
            reason = "No capacity: "
            if current_exposure_usd >= max_total_usd:
                reason += f"total exposure at/above {self.limits.max_total_exposure_pct:.0f}% cap"
            elif portfolio.cash_usd <= 0:
                reason += "insufficient cash"
            else:
                reason += "position size floored to zero after limits applied"
            decision = RiskDecision(approved=False, reason=reason)
            self._log(signal, decision)
            return decision

        shrunk = approved_size < requested_size_usd
        reason = "Approved" if not shrunk else (
            f"Approved at reduced size ${approved_size:,.0f} "
            f"(requested ${requested_size_usd:,.0f}) due to position/exposure limits"
        )
        decision = RiskDecision(approved=True, reason=reason, approved_size_usd=round(approved_size, 2))
        self._log(signal, decision)
        return decision

    def _log(self, signal: Signal, decision: RiskDecision) -> None:
        self.decision_log.append(
            {
                "ticker": signal.ticker,
                "signal_id": signal.signal_id,
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "approved": decision.approved,
                "reason": decision.reason,
                "approved_size_usd": decision.approved_size_usd,
            }
        )
