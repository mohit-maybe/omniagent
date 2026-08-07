"""
omnimarket.ingestion.sentiment_scorer
----------------------------------------
Turns a raw headline (or article body) into a sentiment_score in [-1, 1]
for the omnimarket.models.NewsItem the rest of the engine expects.

Two backends:
  - VaderSentimentScorer (default): VADER, a lexicon/rule-based scorer
    tuned for short, informal text like headlines and social posts. Ships
    its lexicon inside the package - no model download, no GPU, works
    offline once installed. Free and fast; this is what runs unless you
    configure otherwise.
  - TransformerSentimentScorer (optional): wraps a HuggingFace pipeline
    (default: distilbert-base-uncased-finetuned-sst-2-english, matching
    the original spec) for higher accuracy at the cost of a model
    download and heavier runtime dependencies (torch/transformers). Only
    imported if you actually construct this class, so it's not a hard
    dependency for people who just want VADER.

Both implement the same `score(text: str) -> float` interface so
pipeline.py doesn't care which one is wired in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER's stock lexicon is tuned for general/social-media text and misreads
# financial terminology - e.g. it scores "beats earnings expectations" as
# neutral and doesn't recognize "plunge"/"guidance cut" at all. This
# extension follows VADER's own -4..+4 valence scale and is layered on top
# of (not replacing) the base lexicon via update(), a pattern also used by
# finance-tuned VADER variants (e.g. FinVADER) rather than a bespoke scheme.
FINANCE_LEXICON: dict[str, float] = {
    # earnings / guidance
    "beats": 2.2, "beat": 2.2, "misses": -2.2, "miss": -2.0, "missed": -2.2,
    "outperforms": 2.3, "underperforms": -2.3, "raises guidance": 2.5,
    "cuts guidance": -2.5, "lowers guidance": -2.4, "raises forecast": 2.3,
    "cuts forecast": -2.3, "profit warning": -2.8, "guidance": 0.0,

    # analyst actions
    "upgraded": 2.0, "upgrade": 2.0, "downgraded": -2.0, "downgrade": -2.0,
    "overweight": 1.5, "underweight": -1.5, "outperform": 1.8, "buy rating": 2.0,
    "sell rating": -2.0, "price target raised": 2.2, "price target cut": -2.2,

    # price/market action
    "plunge": -3.0, "plunges": -3.0, "plunged": -3.0, "surge": 2.8, "surges": 2.8,
    "surged": 2.8, "rally": 2.3, "rallies": 2.3, "soar": 2.8, "soars": 2.8,
    "soared": 2.8, "tumble": -2.6, "tumbles": -2.6, "tumbled": -2.6,
    "crash": -3.2, "crashes": -3.2, "crashed": -3.2, "rebound": 1.8, "rebounds": 1.8,
    "slump": -2.2, "slumps": -2.2, "sink": -2.0, "sinks": -2.0, "sank": -2.0,
    "rocket": 2.6, "rockets": 2.6,

    # sentiment / regime words
    "bullish": 2.4, "bearish": -2.4, "hawkish": -1.2, "dovish": 1.2,
    "recession": -2.6, "inflation": -0.8, "rate cut": 1.8, "rate hike": -1.5,
    "black swan": -3.0, "default": -2.8, "bankruptcy": -3.2, "bankrupt": -3.2,
    "insolvent": -3.0,

    # regulatory / legal
    "regulatory scrutiny": -1.8, "investigation": -1.8, "lawsuit": -1.8,
    "fine": -1.5, "fined": -1.5, "sanctions": -2.0, "antitrust": -1.5,
    "settlement": -0.5, "approval": 1.8, "approved": 1.8, "cleared": 1.2,
    "rejected": -1.8, "denied": -1.5,

    # product/deal
    "breakthrough": 2.5, "recall": -2.0, "recalls": -2.0, "delay": -1.2,
    "delays": -1.2, "delayed": -1.2, "acquisition": 1.0, "merger": 0.8,
    "partnership": 1.3, "expansion": 1.2, "layoffs": -2.0, "layoff": -2.0,
    "hiring freeze": -1.5,
}


class SentimentScorer(ABC):
    @abstractmethod
    def score(self, text: str) -> float:
        """Return a sentiment score in [-1.0 (bearish), 1.0 (bullish)]."""
        raise NotImplementedError

    def score_batch(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]


class VaderSentimentScorer(SentimentScorer):
    """
    Default scorer. No API key, no model download, no network needed.
    Extends VADER's general-purpose lexicon with finance terminology (see
    FINANCE_LEXICON above). Single-word terms are merged directly into
    VADER's lexicon dict. Multi-word phrases ("raises guidance", "price
    target cut", etc.) are handled separately: VADER tokenizes and scores
    word-by-word, so a phrase key in the lexicon dict alone is silently
    never matched. Instead, phrases are substituted for a placeholder
    single-token before scoring, and that placeholder carries the valence.
    """

    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()

        word_terms = {k: v for k, v in FINANCE_LEXICON.items() if " " not in k}
        phrase_terms = {k: v for k, v in FINANCE_LEXICON.items() if " " in k}

        self._analyzer.lexicon.update(word_terms)

        # Longest phrase first, so e.g. "raises guidance" is matched before
        # any shorter substring could partially interfere.
        self._phrase_placeholders: list[tuple[str, str]] = []  # (phrase_lower, placeholder_token)
        for i, phrase in enumerate(sorted(phrase_terms, key=len, reverse=True)):
            placeholder = f"financeterm{i}"
            self._phrase_placeholders.append((phrase.lower(), placeholder))
            self._analyzer.lexicon[placeholder] = phrase_terms[phrase]

    def score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        processed = text.lower()
        for phrase, placeholder in self._phrase_placeholders:
            if phrase in processed:
                processed = processed.replace(phrase, f" {placeholder} ")
        return float(self._analyzer.polarity_scores(processed)["compound"])


class TransformerSentimentScorer(SentimentScorer):
    """
    Optional higher-accuracy backend matching the spec's original choice
    of distilbert-base-uncased-finetuned-sst-2-english. Requires
    `pip install transformers torch` and a one-time model download from
    huggingface.co (~260MB) - not attempted until you construct this class.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            raise ImportError(
                "TransformerSentimentScorer requires `pip install transformers torch`. "
                "Use VaderSentimentScorer if you don't want that dependency."
            ) from e
        self._pipeline = hf_pipeline("sentiment-analysis", model=model_name)

    def score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        result = self._pipeline(text[:512])[0]  # truncate to the model's typical max length
        magnitude = float(result["score"])  # confidence in [0, 1]
        sign = 1.0 if result["label"].upper().startswith("POS") else -1.0
        return sign * magnitude


def get_default_scorer() -> SentimentScorer:
    return VaderSentimentScorer()
