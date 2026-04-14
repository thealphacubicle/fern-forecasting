"""Clean raw Fern CSVs and write tidy parquet files to ``data/processed/``.

Run from the repo root:

    uv run python src/scripts/clean_raw.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fern_forecasting.preprocessing import (
    clean_calendar,
    clean_inventory,
    clean_orders,
    clean_reviews,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("clean_raw")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CLEANERS = {
    "orders": clean_orders,
    "calendar": clean_calendar,
    "inventory": clean_inventory,
    "reviews": clean_reviews,
}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, cleaner in CLEANERS.items():
        raw_path = RAW_DIR / f"fern_{name}.csv"
        out_path = PROCESSED_DIR / f"{name}_clean.parquet"

        logger.info("loading %s", raw_path)
        raw = pd.read_csv(raw_path)
        cleaned = cleaner(raw)

        logger.info("%s: %d rows -> %d rows", name, len(raw), len(cleaned))
        cleaned.to_parquet(out_path, index=False)
        logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
