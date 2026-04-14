"""Reshape and join cleaned Fern tables into a modeling-ready weekly panel.

The panel is keyed on ``(week_start, product_category)`` at a Monday-anchored
weekly grain, matching the native grain of ``fern_inventory``.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

WEEK_RULE = "W-MON"
LAG_WEEKS = 1
ROLLING_WEEKS = 4


def _to_week_start(dates: pd.Series) -> pd.Series:
    """Snap a datetime series back to the Monday that starts its week."""
    return (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.normalize()


def aggregate_orders_weekly(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level orders to weekly × product_category.

    Args:
        orders: Cleaned orders dataframe from :func:`preprocessing.clean_orders`.

    Returns:
        Dataframe with columns ``week_start``, ``product_category``,
        ``quantity_sold``, ``revenue``, ``order_count``, ``distinct_occasions``.
    """
    df = orders.copy()
    df["week_start"] = _to_week_start(df["order_date"])
    agg = (
        df.groupby(["week_start", "product_category"], observed=True)
        .agg(
            quantity_sold=("quantity_sold", "sum"),
            revenue=("revenue", "sum"),
            order_count=("order_id", "count"),
            distinct_occasions=("occasion_tag", "nunique"),
        )
        .reset_index()
    )
    return agg


def aggregate_calendar_weekly(calendar: pd.DataFrame) -> pd.DataFrame:
    """Roll the daily calendar up to a Monday-anchored weekly grain.

    Args:
        calendar: Cleaned calendar dataframe from :func:`preprocessing.clean_calendar`.

    Returns:
        Dataframe with one row per week and weather / holiday / event features.
    """
    df = calendar.copy()
    df["week_start"] = _to_week_start(df["date"])

    agg = (
        df.groupby("week_start", as_index=False)
        .agg(
            is_holiday_week=("is_holiday", "any"),
            holiday_names=("holiday_name", lambda s: ", ".join(sorted(s.dropna().unique())) or pd.NA),
            min_days_to_holiday=("days_until_next_major_holiday", "min"),
            is_university_event_week=("is_university_event_week", "max"),
            avg_temp_f=("avg_temp_f", "mean"),
            total_precipitation_inches=("precipitation_inches", "sum"),
            season=("season", "first"),
        )
    )
    return agg


def build_weekly_panel(
    orders: pd.DataFrame,
    calendar: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full modeling panel keyed on ``(week_start, product_category)``.

    A cartesian spine of all weeks in the inventory table × all product
    categories is created, then orders, inventory, and calendar features are
    left-joined onto it. Missing order aggregates are filled with zero
    (no sales recorded for that week/category).

    Lag and rolling features are computed per product category.

    Args:
        orders: Cleaned orders dataframe.
        calendar: Cleaned calendar dataframe.
        inventory: Cleaned inventory dataframe.

    Returns:
        Modeling-ready panel dataframe.
    """
    orders_weekly = aggregate_orders_weekly(orders)
    calendar_weekly = aggregate_calendar_weekly(calendar)

    inv = inventory.rename(columns={"order_week": "week_start"}).copy()

    weeks = pd.Index(sorted(inv["week_start"].unique()), name="week_start")
    categories = pd.Index(sorted(inv["product_category"].unique()), name="product_category")
    spine = pd.MultiIndex.from_product([weeks, categories]).to_frame(index=False)

    panel = spine.merge(orders_weekly, on=["week_start", "product_category"], how="left")
    for col in ("quantity_sold", "revenue", "order_count", "distinct_occasions"):
        panel[col] = panel[col].fillna(0)

    panel = panel.merge(inv, on=["week_start", "product_category"], how="left")
    panel = panel.merge(calendar_weekly, on="week_start", how="left")

    panel = panel.sort_values(["product_category", "week_start"]).reset_index(drop=True)
    lag_col = f"quantity_sold_lag{LAG_WEEKS}"
    roll_col = f"quantity_sold_roll{ROLLING_WEEKS}_mean"
    panel[lag_col] = panel.groupby("product_category", observed=True)["quantity_sold"].shift(LAG_WEEKS)
    panel[roll_col] = panel.groupby("product_category", observed=True)[lag_col].transform(
        lambda s: s.rolling(ROLLING_WEEKS, min_periods=1).mean()
    )

    panel["year"] = panel["week_start"].dt.year
    panel["week_of_year"] = panel["week_start"].dt.isocalendar().week.astype("int64")

    logger.info(
        "panel: %d rows across %d weeks × %d categories",
        len(panel),
        panel["week_start"].nunique(),
        panel["product_category"].nunique(),
    )
    return panel
