"""Precompute VADER sentiment for reviews and write ``reviews_scored.parquet``.

Run once (and after any change to the raw reviews) so the Streamlit
dashboard doesn't have to download the NLTK lexicon at boot time.

Run from the repo root::

    uv run python src/scripts/score_reviews.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("score_reviews")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    analyzer = SentimentIntensityAnalyzer()
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    reviews["sentiment"] = reviews["review_text_clean"].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    out = PROCESSED_DIR / "reviews_scored.parquet"
    reviews.to_parquet(out, index=False)
    logger.info("wrote %s (%d rows)", out, len(reviews))


if __name__ == "__main__":
    main()
