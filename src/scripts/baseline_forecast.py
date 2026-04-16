"""Fit and evaluate baseline demand forecasting models.

Reads ``data/processed/weekly_panel.parquet``, fits a linear regression and
a gradient-boosted tree on 2023 data, and reports MAE / RMSE on the 2024
hold-out against a naive lag-1 baseline.

Run from the repo root:

    uv run python src/scripts/baseline_forecast.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fern_forecasting.models import (
    TARGET,
    evaluate,
    fit_baselines,
    prepare_panel,
    split_x_y,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("baseline_forecast")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "weekly_panel.parquet")
    train_df, test_df = prepare_panel(panel)
    logger.info("train: %d rows, test: %d rows", len(train_df), len(test_df))

    x_train, y_train = split_x_y(train_df)
    x_test, _ = split_x_y(test_df)

    models = fit_baselines(x_train, y_train)
    results = evaluate(models, x_test, test_df[TARGET], test_df["product_category"])

    pd.options.display.float_format = "{:.2f}".format

    overall = results[results["category"] == "ALL"].drop(columns="category")
    print("\n=== Overall (test: 2024) ===")
    print(overall.to_string(index=False))

    per_cat = results[results["category"] != "ALL"]
    mae_table = per_cat.pivot(index="category", columns="model", values="mae")
    rmse_table = per_cat.pivot(index="category", columns="model", values="rmse")

    print("\n=== MAE per category ===")
    print(mae_table.to_string())
    print("\n=== RMSE per category ===")
    print(rmse_table.to_string())


if __name__ == "__main__":
    main()
