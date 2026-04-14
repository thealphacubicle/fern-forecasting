"""Build the weekly modeling panel from cleaned Fern tables.

Reads the cleaned parquet files from ``data/processed/``, joins them into a
weekly × product_category panel, and writes the result to
``data/processed/weekly_panel.parquet``.

Run from the repo root:

    uv run python src/scripts/build_panel.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fern_forecasting.features import build_weekly_panel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_panel")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    orders = pd.read_parquet(PROCESSED_DIR / "orders_clean.parquet")
    calendar = pd.read_parquet(PROCESSED_DIR / "calendar_clean.parquet")
    inventory = pd.read_parquet(PROCESSED_DIR / "inventory_clean.parquet")

    panel = build_weekly_panel(orders, calendar, inventory)

    out_path = PROCESSED_DIR / "weekly_panel.parquet"
    panel.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows, %d cols)", out_path, len(panel), panel.shape[1])


if __name__ == "__main__":
    main()
