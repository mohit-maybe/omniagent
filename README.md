# OmniMarket AI — Prediction & Trade-Signal Engine

The backend "brain" of OmniMarket AI: turns price data + news sentiment into
ranked, confidence-scored trade signals, runs them through a risk manager,
and executes/tracks paper trades. This is the piece the CRM dashboard will
call. No real money, no real brokerage connection — paper-trading simulation
only, not financial advice.

## Files

| File | Responsibility |
|---|---|
| `models.py` | Shared dataclasses: `Signal`, `Trade`, `Position`, `PortfolioState`, `RiskEvent`, etc. |
| `indicators.py` | Pure TA functions on OHLCV DataFrames: RSI, MACD, SMA, ATR. |
| `sentiment.py` | Aggregates raw `NewsItem`s into per-ticker sentiment momentum (`SentimentAggregator`). |
| `prediction_engine.py` | Combines technicals + sentiment + regime detection into a `Signal` (direction, 0–100 confidence, plain-English rationale). Transparent weighted scoring, not a black box — swap in a trained model later without changing the interface. |
| `risk_manager.py` | Enforces position/exposure caps and blocks trades during flagged high-risk events. Every decision is logged. |
| `trade_engine.py` | Opens/closes simulated positions against a virtual `PortfolioState`, sets ATR-based stop-loss/take-profit, logs every trade with its rationale. On close, hands the outcome to the learning engine if one is attached. |
| `learning_engine.py` | **Learns from the engine's mistakes.** Logs the component scores (sentiment/technical/regime) behind every closed trade along with win/loss. Once enough labeled trades accumulate, fits a logistic regression and reweights `PredictionEngine` toward whichever component has actually been predictive — persisted to disk so it survives restarts. |
| `api.py` | FastAPI layer (`/signals`, `/ingest`, `/trades`, `/portfolio`, `/risk-events`, `/mark-to-market`, `/learning/*`) for the dashboard to call. |
| `scheduler.py` | Runs the full ingest→signal→risk→trade cycle on a timer (default: daily briefing at 00:00 UTC + mark-to-market every 5 min), matching the spec. |
| `ingestion/config.py` | Loads API keys from environment / `.env`. Missing keys are fine at import time - a source only errors when actually used. |
| `ingestion/rate_limiter.py` | Sliding-window rate limiters so batch jobs respect each free-tier quota (Marketaux 100/day, Finnhub 60/min, etc). |
| `ingestion/price_sources.py` | Real price clients: `AlpacaPriceSource` + `AlphaVantagePriceSource` (stocks), `BinancePriceSource` + `CoinCapPriceSource` (crypto, no key needed). |
| `ingestion/news_sources.py` | Real news clients: `MarketauxNewsSource` + `FinnhubNewsSource`, with automatic event-type classification (earnings/regulation/macro/general). |
| `ingestion/sentiment_scorer.py` | VADER-based scorer extended with a finance-domain lexicon (VADER's general lexicon misreads terms like "beats earnings" or "plunge" out of the box). Optional HuggingFace transformer backend available. |
| `ingestion/pipeline.py` | Orchestrates it all: primary/fallback source chains, rate limiting, graceful degradation, normalization into the exact structures `indicators.py`/`sentiment.py` expect. |
| `demo.py` | End-to-end run against **synthetic** data — proves the whole pipeline without needing any API keys. |

## Run the demo (no API keys needed)

```bash
pip install -r requirements.txt
python demo.py
```

This generates synthetic OHLCV + news for 6 tickers, produces ranked
signals, runs them through the risk manager, opens paper trades, and prints
a daily-briefing-style summary — including a deterministic proof that an
active risk event (e.g. an earnings-gap flag) blocks a trade even at 90%
confidence.

## Run the API

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`. Core flow:

1. `POST /signals` — send OHLCV bars + news items per ticker, get back ranked signals. (Bring-your-own-data.)
1b. `POST /ingest` — same output, but pulls REAL price + news data itself via the ingestion pipeline. Just send `{"tickers": {...}}`.
2. `POST /trades` — attempt to open a paper trade on a ticker from the last computed signal.
3. `POST /mark-to-market` — send current prices, auto-closes any position that hit its stop/target.
4. `GET /portfolio` — current equity, exposure, open positions, full trade log.
5. `POST /risk-events` / `DELETE /risk-events/{ticker}` — flag/clear high-risk conditions (earnings, macro shocks) that block trading.

## Using real data (now implemented)

1. `cp .env.example .env` and fill in whichever API keys you have (all free
   tiers — see `.env.example` for signup links). None are required to boot
   the system: missing keys just mean that source is skipped in favor of
   its fallback, or the ticker is skipped with a clear reason if nothing
   in the chain works.
2. Either:
   - **One-off / on-demand**: `POST /ingest` with `{"tickers": {"AAPL":
     "stock", "BTC": "crypto"}}` — pulls real price + news data and
     returns ranked signals plus per-ticker diagnostics (which source
     served the data, or why it was skipped).
   - **Scheduled**: `python scheduler.py` runs continuously, doing a full
     ingest→signal→risk→trade cycle daily at 00:00 UTC (configurable) plus
     a mark-to-market check every 5 minutes. Use `python scheduler.py
     --once` to run a single cycle immediately instead of starting the
     long-running scheduler (useful for testing or wiring into your own
     cron).

**Source chains** (primary → fallback, first one with a working key wins):
- Stocks: Alpaca → Alpha Vantage
- Crypto: Binance → CoinCap (neither needs a key — public market data)
- News: Marketaux → Finnhub

**A note on this sandbox specifically**: the environment this was built in
has network egress locked to package registries only, so the ingestion
code has been verified by parsing real API response shapes against mocked
HTTP calls (`unittest.mock`), not by an actual live call to Alpaca/Binance/
Marketaux. Run it somewhere with open network access and real keys for the
first live test — the parsing logic has been checked against each
provider's documented response format, but a live smoke test is still
worth doing before relying on it.

**Known limitation — CoinCap OHLCV is approximated.** CoinCap's free
`/history` endpoint returns price *points*, not true OHLC candles, and
doesn't expose volume. `CoinCapPriceSource` sets open=high=low=close to
the point price and volume=0. This is fine as a last-resort fallback but
will understate ATR (and therefore stop-loss/take-profit distance) versus
a real OHLCV source — avoid relying on CoinCap as your primary crypto
source if you can get a Binance-reachable deployment instead.

**Persistence**: `PortfolioState` and `RiskManager.decision_log` are
in-memory in `api.py`. Swap for SQLite or Supabase before deploying so
state survives a restart (the learning engine's weights already persist
to disk independently — see below).

## The learning system

The engine's confidence score is a weighted blend of three components —
sentiment momentum, technical indicators, and market regime. Those weights
start at fixed defaults (40/45/15), but they don't have to stay fixed:

1. Every time a signal leads to a trade, its raw component scores get
   attached to that `Trade` (`signal_components`).
2. When the trade closes — win, loss, or manual close —
   `PaperTradingEngine` automatically hands the outcome to
   `AdaptiveLearningEngine.record_outcome()`, which logs
   `{sentiment, technical, regime} -> win/loss` to `learning_store.json`.
3. Once at least 20 closed trades are logged (with both wins and losses
   represented), `retrain()` fits a 3-feature logistic regression predicting
   win probability from the component scores. The regression coefficients
   tell you, in plain terms, which component has actually been predictive.
4. Those coefficients are converted into new `PredictionEngine` weights
   (weighted by *magnitude* — a component the model leans on heavily, in
   either direction, gets more trust; a component near-zero gets
   down-weighted) and persisted to `learned_weights.json`.
5. On the next API startup, the persisted weights are loaded automatically
   — the engine remembers what it learned.

Run `python demo.py` to see this end-to-end: it simulates 60 real
open→close trades where the technical component is deliberately made
predictive and sentiment is pure noise, retrains, and prints the weight
shift (technical rises, sentiment falls toward its floor).

Via the API:

```bash
GET  /learning/status     # closed-trade count, wins/losses, current weights, training history
POST /learning/retrain    # retrain now if enough data has accumulated (add ?force=true to override the minimum)
```

**Why logistic regression and not a deep model:** with realistic trade
volumes (dozens to low hundreds of closed trades, not millions), a 3-feature
linear model is what can actually be fit reliably, and its coefficients are
explainable in one sentence — which matters for the audit trail this system
is built around. Swap in a richer model later without changing the
`retrain()` / `get_weights()` / `set_weights()` interface.

**Guardrails:** a floor (5% per component) keeps no single input from being
zeroed out by one unlucky batch of trades, and retraining requires both
outcome classes present (not just wins or just losses) so it can't overfit
to a lucky or unlucky streak.

## Tunable parameters

- `PredictionEngine(weights=EngineWeights(sentiment=..., technical=..., regime=...))`
  — relative weight of each input (defaults 40/45/15, normalized).
- `RiskLimits(max_position_pct=10, max_total_exposure_pct=50, min_confidence_to_trade=55)`
  — position sizing and the minimum confidence required to trade at all.
- `ATRStopConfig(stop_loss_atr_mult=1.5, take_profit_atr_mult=3.0)` — stop/target
  distance as a multiple of Average True Range (~2:1 reward/risk by default).

## Disclaimer

This is a paper-trading simulation for educational/research purposes. It is
not financial advice, and past performance in simulation does not indicate
future results in live markets.
