"""Tune HGBT, fit a per-category rose model, and render diagnostic plots.

Run from the repo root:

    uv run python src/scripts/tune_and_plot.py

Writes three PNGs to ``figures/``:
- ``baseline_pred_vs_actual.png`` — tuned HGBT predicted vs actual (2024)
- ``roses_2024_timeseries.png`` — actual vs global vs per-category for roses
- ``baseline_feature_importance.png`` — permutation importance on 2024 test set
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fern_forecasting.models import (
    ALL_FEATURES,
    TARGET,
    evaluate,
    fit_per_category_hgbt,
    prepare_panel,
    split_x_y,
    tune_hgbt,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tune_and_plot")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "figures"


def _plot_pred_vs_actual(test_df: pd.DataFrame, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    categories = test_df["product_category"].unique()
    for cat in sorted(categories):
        mask = (test_df["product_category"] == cat).to_numpy()
        ax.scatter(test_df.loc[mask, TARGET], y_pred[mask], s=22, alpha=0.7, label=cat)
    lim = max(float(test_df[TARGET].max()), float(np.max(y_pred))) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual quantity_sold")
    ax.set_ylabel("Predicted quantity_sold")
    ax.set_title("Tuned HGBT: predicted vs actual (2024 hold-out)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roses_timeseries(
    test_df: pd.DataFrame,
    y_global: np.ndarray,
    y_per_cat: np.ndarray,
    path: Path,
) -> None:
    mask = (test_df["product_category"] == "roses").to_numpy()
    weeks = pd.to_datetime(test_df.loc[mask, "week_start"])
    actual = test_df.loc[mask, TARGET].to_numpy()
    order = np.argsort(weeks.values)
    weeks_sorted = weeks.values[order]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weeks_sorted, actual[order], marker="o", label="actual", color="black")
    ax.plot(weeks_sorted, y_global[mask][order], marker="s", label="global HGBT (tuned)", alpha=0.8)
    ax.plot(weeks_sorted, y_per_cat[order], marker="^", label="per-category HGBT", alpha=0.8)
    ax.set_title("Roses — weekly demand, 2024 hold-out")
    ax.set_ylabel("quantity_sold")
    ax.set_xlabel("week_start")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(importances: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Mean decrease in R² (permutation importance)")
    ax.set_title("Tuned HGBT — feature importance on 2024 hold-out")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PROCESSED_DIR / "weekly_panel.parquet")
    train_df, test_df = prepare_panel(panel)

    tuned_global, best_params, best_cv_rmse = tune_hgbt(train_df)
    logger.info("best params: %s (CV log-RMSE=%.4f)", best_params, best_cv_rmse)

    x_test, _ = split_x_y(test_df)
    y_test_raw = test_df[TARGET].to_numpy(dtype=float)

    y_global_pred = np.clip(np.expm1(tuned_global.predict(x_test)), 0.0, None)

    rose_model = fit_per_category_hgbt(train_df, "roses", **{
        k.replace("reg__", ""): v for k, v in best_params.items()
    })
    rose_mask = (test_df["product_category"] == "roses").to_numpy()
    rose_x_test = x_test[rose_mask]
    y_per_cat_rose = np.clip(np.expm1(rose_model.predict(rose_x_test)), 0.0, None)

    per_cat_full = y_global_pred.copy()
    per_cat_full[rose_mask] = y_per_cat_rose

    results = evaluate(
        {"hgbt_tuned": tuned_global},
        x_test,
        test_df[TARGET],
        test_df["product_category"],
    )
    print("\n=== Tuned HGBT (test: 2024) ===")
    print(results[results["category"] == "ALL"].to_string(index=False))

    rose_actual = y_test_raw[rose_mask]
    rose_global_mae = float(np.mean(np.abs(rose_actual - y_global_pred[rose_mask])))
    rose_global_rmse = float(np.sqrt(np.mean((rose_actual - y_global_pred[rose_mask]) ** 2)))
    rose_pc_mae = float(np.mean(np.abs(rose_actual - y_per_cat_rose)))
    rose_pc_rmse = float(np.sqrt(np.mean((rose_actual - y_per_cat_rose) ** 2)))
    print("\n=== Roses-only comparison (2024) ===")
    print(f"global HGBT (tuned):  MAE={rose_global_mae:.2f}  RMSE={rose_global_rmse:.2f}")
    print(f"per-category HGBT:    MAE={rose_pc_mae:.2f}  RMSE={rose_pc_rmse:.2f}")

    perm = permutation_importance(
        tuned_global, x_test, np.log1p(y_test_raw), n_repeats=10, random_state=0, n_jobs=-1
    )
    importances = pd.Series(perm.importances_mean, index=ALL_FEATURES)

    _plot_pred_vs_actual(test_df, y_global_pred, FIGURES_DIR / "baseline_pred_vs_actual.png")
    _plot_roses_timeseries(test_df, y_global_pred, y_per_cat_rose, FIGURES_DIR / "roses_2024_timeseries.png")
    _plot_feature_importance(importances, FIGURES_DIR / "baseline_feature_importance.png")
    logger.info("wrote plots to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
