"""Baseline demand forecasting models.

Fits a linear regression and a gradient-boosted tree on the weekly panel
produced by :mod:`fern_forecasting.features`, and evaluates both against a
naive lag-1 baseline. Target is ``log1p(quantity_sold)``; predictions are
back-transformed with ``expm1`` and clipped at zero before scoring.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

TARGET = "quantity_sold"

CATEGORICAL_FEATURES: list[str] = ["product_category", "season"]
NUMERIC_FEATURES: list[str] = [
    "week_of_year",
    "min_days_to_holiday",
    "avg_temp_f",
    "total_precipitation_inches",
    "quantity_sold_lag1",
    "quantity_sold_roll4_mean",
    "unit_cost",
    "unit_price",
]
BOOLEAN_FEATURES: list[str] = ["is_holiday_week", "is_university_event_week"]
ALL_FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES


def prepare_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the panel and split it into 2023 train / 2024 test frames.

    Drops rows where lag/rolling features are undefined (first week per
    category) and coerces boolean features to integers.

    Args:
        panel: Weekly panel produced by :func:`features.build_weekly_panel`.

    Returns:
        Tuple of ``(train_df, test_df)`` with all modeling columns retained.
    """
    df = panel.dropna(subset=["quantity_sold_lag1", "quantity_sold_roll4_mean"]).copy()
    df[BOOLEAN_FEATURES] = df[BOOLEAN_FEATURES].astype(int)

    train_df = df[df["year"] <= 2023].reset_index(drop=True)
    test_df = df[df["year"] == 2024].reset_index(drop=True)
    return train_df, test_df


def split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and log-transformed target from a panel slice."""
    return df[ALL_FEATURES], np.log1p(df[TARGET])


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ]
    )


def fit_baselines(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Fit a linear regression and a gradient-boosted tree on log-target.

    Args:
        x_train: Training feature dataframe.
        y_train: Log-transformed training target.

    Returns:
        Dict mapping model name to fitted sklearn :class:`Pipeline`.
    """
    models: dict[str, Pipeline] = {
        "linear": Pipeline([("pre", _preprocessor()), ("reg", LinearRegression())]),
        "hgbt": _hgbt_pipeline(),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        logger.info("fitted %s", name)
    return models


def _hgbt_pipeline(**reg_kwargs: object) -> Pipeline:
    return Pipeline(
        [
            ("pre", _preprocessor()),
            ("reg", HistGradientBoostingRegressor(random_state=0, max_iter=300, **reg_kwargs)),
        ]
    )


def tune_hgbt(
    train_df: pd.DataFrame, n_splits: int = 3
) -> tuple[Pipeline, dict[str, object], float]:
    """Grid-search HGBT hyperparameters using time-series CV within 2023.

    Rows are sorted by ``week_start`` before splitting so each fold respects
    chronological order.

    Args:
        train_df: Training slice returned by :func:`prepare_panel`.
        n_splits: Number of time-series CV folds.

    Returns:
        Tuple of (best fitted pipeline, best params, best CV RMSE on log-target).
    """
    sorted_df = train_df.sort_values("week_start").reset_index(drop=True)
    x_train, y_train = split_x_y(sorted_df)

    grid = {
        "reg__learning_rate": [0.05, 0.1],
        "reg__max_depth": [3, 5, None],
        "reg__min_samples_leaf": [10, 20],
    }
    search = GridSearchCV(
        _hgbt_pipeline(),
        grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    logger.info("tuned hgbt: best CV RMSE (log) = %.4f, params = %s", -search.best_score_, search.best_params_)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


def fit_per_category_hgbt(
    train_df: pd.DataFrame, category: str, **reg_kwargs: object
) -> Pipeline:
    """Fit an HGBT pipeline on a single product category's training rows.

    Args:
        train_df: Training slice returned by :func:`prepare_panel`.
        category: Product category label to filter on.
        **reg_kwargs: Extra keyword args passed to :class:`HistGradientBoostingRegressor`.

    Returns:
        Fitted pipeline.
    """
    sub = train_df[train_df["product_category"] == category]
    x, y = split_x_y(sub)
    pipeline = _hgbt_pipeline(**reg_kwargs)
    pipeline.fit(x, y)
    logger.info("fitted per-category hgbt for %s on %d rows", category, len(sub))
    return pipeline


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate(
    models: dict[str, Pipeline],
    x_test: pd.DataFrame,
    y_test_raw: pd.Series,
    categories: pd.Series,
) -> pd.DataFrame:
    """Score fitted models plus a naive lag-1 baseline per category.

    Predictions are back-transformed with ``expm1`` and clipped at zero before
    comparison against raw ``quantity_sold``.

    Args:
        models: Dict of fitted pipelines.
        x_test: Test feature dataframe.
        y_test_raw: Raw (non-log) test target series.
        categories: ``product_category`` series aligned to ``x_test``.

    Returns:
        Long dataframe with columns ``model``, ``category``, ``mae``, ``rmse``.
        ``category == "ALL"`` rows hold the overall scores.
    """
    y_true = y_test_raw.to_numpy(dtype=float)
    predictions: dict[str, np.ndarray] = {
        name: np.clip(np.expm1(model.predict(x_test)), 0.0, None)
        for name, model in models.items()
    }
    predictions["naive_lag1"] = np.clip(x_test["quantity_sold_lag1"].to_numpy(dtype=float), 0.0, None)

    cats = categories.to_numpy()
    rows: list[dict[str, object]] = []
    for name, y_pred in predictions.items():
        rows.append({"model": name, "category": "ALL", **_score(y_true, y_pred)})
        for cat in np.unique(cats):
            mask = cats == cat
            rows.append({"model": name, "category": str(cat), **_score(y_true[mask], y_pred[mask])})

    return pd.DataFrame(rows).sort_values(["model", "category"]).reset_index(drop=True)
