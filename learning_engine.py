"""
omnimarket.learning_engine
-----------------------------
Closes the loop between "a trade happened" and "the engine gets better".

How it learns:
  1. Every closed trade (win or loss) is logged with the raw component
     scores that produced its signal - sentiment score, technical score,
     regime bias - plus the outcome label (1 = win, 0 = loss/breakeven).
  2. Once enough labeled trades have accumulated, `retrain()` fits a
     logistic regression: P(win) ~ sentiment, technical, regime.
  3. The learned coefficients are converted into new PredictionEngine
     weights (bigger |coefficient| = that component has actually been
     more predictive of wins historically -> gets more weight next time;
     a component with a *negative* coefficient - i.e. historically
     anti-predictive - gets down-weighted, not just re-signed, since the
     weighting scheme is about how much to trust a component, not its
     directionality).
  4. Weights are persisted to disk (JSON) so learning survives restarts,
     along with a training log (accuracy, sample size, timestamp) for
     auditability - the "did the model actually get better" question
     should always be answerable, not just asserted.

This is intentionally a simple, inspectable model (3-feature logistic
regression) rather than a deep model - given the ethics_reminder in this
system prompt about explainable AI and given >100 trades is a realistic
amount of feedback to expect early on, a model this size can actually be
fit reliably and its coefficients can be explained in one sentence.
Swap in a richer model later; keep the retrain()/get_weights() interface.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from models import Trade
from prediction_engine import EngineWeights

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - keeps the module importable without sklearn
    _SKLEARN_AVAILABLE = False


DEFAULT_STORE_PATH = os.path.join(os.path.dirname(__file__), "learning_store.json")
DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "learned_weights.json")

MIN_SAMPLES_TO_RETRAIN = 20   # need at least this many closed trades before trusting a fit
MIN_SAMPLES_PER_CLASS = 5     # need at least this many wins AND losses, or the fit is meaningless


@dataclass
class OutcomeRecord:
    trade_id: str
    ticker: str
    sentiment_component: float
    technical_component: float
    regime_component: float
    confidence: float
    direction: str
    win: int              # 1 = win, 0 = loss/breakeven
    pnl_usd: float
    roi_pct: float
    closed_at: str


@dataclass
class TrainingResult:
    trained_at: str
    n_samples: int
    n_wins: int
    n_losses: int
    train_accuracy: float
    weights: dict           # {"sentiment": .., "technical": .., "regime": ..}
    coefficients: dict      # raw logistic-regression coefficients, for auditability
    note: str = ""


class AdaptiveLearningEngine:
    """
    Usage:
        learning_engine = AdaptiveLearningEngine()
        trader = PaperTradingEngine(portfolio, risk_manager, learning_engine=learning_engine)
        # ... trades open and close normally; each close calls record_outcome() automatically ...
        result = learning_engine.maybe_retrain(prediction_engine)  # call periodically (e.g. daily)
    """

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        min_samples_to_retrain: int = MIN_SAMPLES_TO_RETRAIN,
    ):
        self.store_path = store_path
        self.weights_path = weights_path
        self.min_samples_to_retrain = min_samples_to_retrain
        self.training_log: list[TrainingResult] = []
        self._load_training_log()

    # ---------------- outcome logging ----------------

    def record_outcome(self, trade: Trade) -> None:
        """Called automatically by PaperTradingEngine whenever a trade closes."""
        if not trade.signal_components:
            return  # trade predates the components field, or was opened without a signal - skip
        if trade.status not in ("closed_win", "closed_loss", "closed_manual"):
            return

        win = 1 if (trade.pnl_usd or 0) > 0 else 0

        record = OutcomeRecord(
            trade_id=trade.trade_id,
            ticker=trade.ticker,
            sentiment_component=trade.signal_components.get("sentiment", 0.0),
            technical_component=trade.signal_components.get("technical", 0.0),
            regime_component=trade.signal_components.get("regime", 0.0),
            confidence=trade.confidence,
            direction=trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
            win=win,
            pnl_usd=trade.pnl_usd or 0.0,
            roi_pct=trade.roi_pct or 0.0,
            closed_at=trade.closed_at or datetime.now(timezone.utc).isoformat(),
        )
        self._append_record(record)

    def _append_record(self, record: OutcomeRecord) -> None:
        records = self._load_records()
        records.append(asdict(record))
        with open(self.store_path, "w") as f:
            json.dump(records, f, indent=2)

    def _load_records(self) -> list[dict]:
        if not os.path.exists(self.store_path):
            return []
        with open(self.store_path) as f:
            return json.load(f)

    # ---------------- training ----------------

    def sample_counts(self) -> tuple[int, int, int]:
        records = self._load_records()
        wins = sum(1 for r in records if r["win"] == 1)
        losses = len(records) - wins
        return len(records), wins, losses

    def maybe_retrain(self, prediction_engine=None) -> Optional[TrainingResult]:
        """
        Retrain if there's enough labeled data, and (if a PredictionEngine is
        passed) push the new weights into it immediately. Returns None if
        there wasn't enough data to retrain yet - callers should treat that
        as "still collecting feedback", not an error.
        """
        n, wins, losses = self.sample_counts()
        if n < self.min_samples_to_retrain or wins < MIN_SAMPLES_PER_CLASS or losses < MIN_SAMPLES_PER_CLASS:
            return None
        return self.retrain(prediction_engine)

    def retrain(self, prediction_engine=None) -> TrainingResult:
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for retrain() - pip install scikit-learn")

        records = self._load_records()
        if len(records) < 4:
            raise RuntimeError(f"Not enough closed trades to retrain (have {len(records)}, need >= 4)")

        X = np.array([[r["sentiment_component"], r["technical_component"], r["regime_component"]] for r in records])
        y = np.array([r["win"] for r in records])

        if len(set(y.tolist())) < 2:
            raise RuntimeError("Cannot retrain: all logged outcomes are the same class (all wins or all losses)")

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        train_accuracy = float(model.score(X, y))

        coef = model.coef_[0]  # [sentiment, technical, regime]
        coefficients = {"sentiment": float(coef[0]), "technical": float(coef[1]), "regime": float(coef[2])}

        new_weights = self._coefficients_to_weights(coefficients)

        result = TrainingResult(
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_samples=len(records),
            n_wins=int(y.sum()),
            n_losses=int(len(y) - y.sum()),
            train_accuracy=round(train_accuracy, 4),
            weights=asdict(new_weights) if not isinstance(new_weights, dict) else new_weights,
            coefficients=coefficients,
        )

        self._save_weights(new_weights, result)
        self.training_log.append(result)
        self._save_training_log()

        if prediction_engine is not None:
            prediction_engine.set_weights(new_weights)

        return result

    @staticmethod
    def _coefficients_to_weights(coefficients: dict) -> EngineWeights:
        """
        Convert logistic-regression coefficients into engine weights.
        We weight by MAGNITUDE (how strongly a component's score moves the
        predicted win probability, in either direction) rather than raw
        signed coefficient, because the engine weights control *how much
        to trust* a component's contribution to the composite score - the
        component's own sign already encodes bullish/bearish. A component
        whose coefficient magnitude is near zero has not, historically,
        been predictive of outcomes and should be down-weighted regardless
        of sign.
        A small floor (5% each) keeps any single component from being
        zeroed out entirely on limited data - full removal should require
        sustained evidence across many retrains, not one unlucky sample.
        """
        floor = 0.05
        magnitudes = {k: abs(v) for k, v in coefficients.items()}
        total = sum(magnitudes.values())

        if total < 1e-9:
            return EngineWeights()  # no signal in the data yet - keep defaults

        raw = {k: floor + (1 - 3 * floor) * (m / total) for k, m in magnitudes.items()}
        return EngineWeights(sentiment=raw["sentiment"], technical=raw["technical"], regime=raw["regime"]).normalized()

    # ---------------- persistence ----------------

    def _save_weights(self, weights: EngineWeights, result: TrainingResult) -> None:
        payload = {
            "weights": asdict(weights),
            "trained_at": result.trained_at,
            "n_samples": result.n_samples,
            "train_accuracy": result.train_accuracy,
            "coefficients": result.coefficients,
        }
        with open(self.weights_path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_persisted_weights(self) -> Optional[EngineWeights]:
        if not os.path.exists(self.weights_path):
            return None
        with open(self.weights_path) as f:
            payload = json.load(f)
        w = payload["weights"]
        return EngineWeights(sentiment=w["sentiment"], technical=w["technical"], regime=w["regime"])

    def _save_training_log(self) -> None:
        log_path = self.store_path.replace("learning_store.json", "training_log.json")
        with open(log_path, "w") as f:
            json.dump([asdict(r) for r in self.training_log], f, indent=2)

    def _load_training_log(self) -> None:
        log_path = self.store_path.replace("learning_store.json", "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                raw = json.load(f)
            self.training_log = [TrainingResult(**r) for r in raw]
