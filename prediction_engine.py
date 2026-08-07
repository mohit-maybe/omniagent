"""
omnimarket.prediction_engine
------------------------------
Combines:
  1. Sentiment momentum (news velocity + score delta)
  2. Technical indicators (RSI, MACD, moving-average crossover)
  3. Market regime detection (bull / bear / volatile / ranging)

...into a single Signal per ticker: direction (BUY/SELL/HOLD) and a
0-100 confidence score, with a human-readable rationale string suitable
for the dashboard's "Prediction Calendar" and trade journal.

This is a transparent, rule-based hybrid model (weighted scoring), not a
black box - every signal's rationale can be traced back to the inputs
that produced it, which matters for the "explainable AI" requirement.
Swap `PredictionEngine._score_technical` / `_score_sentiment` internals
for a trained LSTM/XGBoost model later without changing the interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from models import (
    AssetClass,
    Direction,
    MarketRegime,
    Signal,
    SentimentSnapshot,
    TechnicalSnapshot,
)


@dataclass
class EngineWeights:
    """Relative weight of each signal component in the final confidence score."""
    sentiment: float = 0.4
    technical: float = 0.45
    regime: float = 0.15

    def normalized(self) -> "EngineWeights":
        total = self.sentiment + self.technical + self.regime
        return EngineWeights(self.sentiment / total, self.technical / total, self.regime / total)


class PredictionEngine:
    def __init__(self, weights: EngineWeights | None = None):
        self.weights = (weights or EngineWeights()).normalized()

    def set_weights(self, weights: EngineWeights) -> None:
        """Used by the learning layer to push in retrained weights."""
        self.weights = weights.normalized()

    def get_weights(self) -> EngineWeights:
        return self.weights

    # ---- component scorers, each returns a value in [-1.0, 1.0] ----

    @staticmethod
    def _score_sentiment(snap: SentimentSnapshot | None) -> float:
        if snap is None or snap.article_count == 0:
            return 0.0
        # Blend absolute sentiment with its momentum (delta), lightly
        # scaled by velocity so a single stale article doesn't dominate.
        momentum = snap.mean_score * 0.7 + snap.score_delta * 0.3
        velocity_factor = min(1.0, 0.3 + snap.velocity / 4.0)  # more articles -> more trust
        return max(-1.0, min(1.0, momentum * velocity_factor))

    @staticmethod
    def _score_technical(tech: TechnicalSnapshot) -> float:
        score = 0.0
        # RSI: overbought/oversold mean-reversion signal
        if tech.rsi <= 30:
            score += 0.5
        elif tech.rsi >= 70:
            score -= 0.5
        else:
            score += (50 - tech.rsi) / 100  # mild pull toward reversion

        # MACD: trend/momentum confirmation
        macd_diff = tech.macd - tech.macd_signal
        score += max(-0.35, min(0.35, macd_diff / max(abs(tech.last_price) * 0.001, 1e-6)))

        # Moving average crossover: trend direction
        if tech.sma_fast > tech.sma_slow:
            score += 0.25
        elif tech.sma_fast < tech.sma_slow:
            score -= 0.25

        return max(-1.0, min(1.0, score))

    @staticmethod
    def detect_regime(tech: TechnicalSnapshot) -> MarketRegime:
        volatility_ratio = tech.atr / tech.last_price if tech.last_price else 0
        trend_strength = (tech.sma_fast - tech.sma_slow) / tech.last_price if tech.last_price else 0

        if volatility_ratio > 0.035:
            return MarketRegime.VOLATILE
        if trend_strength > 0.01:
            return MarketRegime.BULL
        if trend_strength < -0.01:
            return MarketRegime.BEAR
        return MarketRegime.RANGING

    @staticmethod
    def _regime_bias(regime: MarketRegime) -> float:
        return {
            MarketRegime.BULL: 0.3,
            MarketRegime.BEAR: -0.3,
            MarketRegime.VOLATILE: -0.1,   # volatility slightly favors caution
            MarketRegime.RANGING: 0.0,
        }[regime]

    def generate_signal(
        self,
        ticker: str,
        asset_class: AssetClass,
        tech: TechnicalSnapshot,
        sentiment: SentimentSnapshot | None,
    ) -> Signal:
        regime = self.detect_regime(tech)

        s_sent = self._score_sentiment(sentiment)
        s_tech = self._score_technical(tech)
        s_regime = self._regime_bias(regime)

        composite = (
            s_sent * self.weights.sentiment
            + s_tech * self.weights.technical
            + s_regime * self.weights.regime
        )  # in [-1, 1]

        confidence = round(min(100.0, abs(composite) * 100 * 1.15), 1)  # slight scale to use fuller range
        confidence = max(0.0, min(100.0, confidence))

        if composite > 0.12:
            direction = Direction.BUY
        elif composite < -0.12:
            direction = Direction.SELL
        else:
            direction = Direction.HOLD
            confidence = min(confidence, 40.0)  # HOLD signals shouldn't read as high-conviction

        rationale = self._build_rationale(ticker, tech, sentiment, regime, s_sent, s_tech)

        return Signal(
            ticker=ticker,
            asset_class=asset_class,
            direction=direction,
            confidence=confidence,
            regime=regime,
            rationale=rationale,
            components={"sentiment": round(s_sent, 4), "technical": round(s_tech, 4), "regime": round(s_regime, 4)},
        )

    @staticmethod
    def _build_rationale(
        ticker: str,
        tech: TechnicalSnapshot,
        sentiment: SentimentSnapshot | None,
        regime: MarketRegime,
        s_sent: float,
        s_tech: float,
    ) -> str:
        parts = [f"Regime: {regime.value}."]

        if sentiment and sentiment.article_count > 0:
            tone = "bullish" if s_sent > 0.1 else "bearish" if s_sent < -0.1 else "neutral"
            parts.append(
                f"Sentiment {tone} ({sentiment.mean_score:+.2f} avg over "
                f"{sentiment.article_count} articles, delta {sentiment.score_delta:+.2f})."
            )
        else:
            parts.append("No recent news coverage.")

        rsi_note = "oversold" if tech.rsi <= 30 else "overbought" if tech.rsi >= 70 else f"RSI {tech.rsi:.0f}"
        macd_note = "MACD bullish cross" if tech.macd > tech.macd_signal else "MACD bearish cross"
        ma_note = "fast MA > slow MA" if tech.sma_fast > tech.sma_slow else "fast MA < slow MA"
        parts.append(f"{rsi_note}, {macd_note}, {ma_note}.")

        return " ".join(parts)

    def generate_batch(
        self,
        tech_snapshots: dict[str, TechnicalSnapshot],
        sentiment_snapshots: dict[str, SentimentSnapshot],
        asset_classes: dict[str, AssetClass],
    ) -> list[Signal]:
        signals = []
        for ticker, tech in tech_snapshots.items():
            sent = sentiment_snapshots.get(ticker)
            asset_class = asset_classes.get(ticker, AssetClass.STOCK)
            signals.append(self.generate_signal(ticker, asset_class, tech, sent))
        return sorted(signals, key=lambda s: s.confidence, reverse=True)
