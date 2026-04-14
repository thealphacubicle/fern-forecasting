"""Cleaning functions for Fern raw CSVs.

Each ``clean_*`` function takes a raw dataframe (as read from ``data/raw/``)
and returns a tidy, typed dataframe suitable for downstream feature
engineering and modeling.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

HOLIDAY_CAP_DAYS = 90
HOLIDAY_SENTINEL = 999


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw transaction-level orders table.

    Parses ``order_date`` as datetime, normalizes string categoricals, and
    drops exact duplicate rows.

    Args:
        df: Raw dataframe loaded from ``fern_orders.csv``.

    Returns:
        Cleaned copy with typed columns.
    """
    out = df.drop_duplicates().copy()
    out["order_date"] = pd.to_datetime(out["order_date"], errors="raise")
    out["day_of_week"] = out["order_date"].dt.day_name()

    for col in ("product_category", "occasion_tag", "order_channel"):
        out[col] = out[col].astype("string").str.strip().str.lower()

    if (out["quantity_sold"] <= 0).any():
        logger.warning("orders: %d rows with non-positive quantity_sold", (out["quantity_sold"] <= 0).sum())
    if (out["revenue"] < 0).any():
        logger.warning("orders: %d rows with negative revenue", (out["revenue"] < 0).sum())

    return out.sort_values("order_date").reset_index(drop=True)


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the daily calendar table.

    Parses ``date`` as datetime, recodes the 999 ``days_until_next_major_holiday``
    sentinel (capped at ``HOLIDAY_CAP_DAYS``), and adds an ``is_holiday`` flag.

    Args:
        df: Raw dataframe loaded from ``fern_calendar.csv``.

    Returns:
        Cleaned copy with typed columns.
    """
    out = df.drop_duplicates().copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["season"] = out["season"].astype("string").str.strip().str.lower()
    out["holiday_name"] = out["holiday_name"].astype("string").str.strip()
    out["is_holiday"] = out["holiday_name"].notna()

    days = out["days_until_next_major_holiday"]
    sentinel_count = int((days == HOLIDAY_SENTINEL).sum())
    if sentinel_count:
        logger.info("calendar: recoding %d rows with %d sentinel", sentinel_count, HOLIDAY_SENTINEL)
    out["days_until_next_major_holiday"] = days.where(days != HOLIDAY_SENTINEL, HOLIDAY_CAP_DAYS).clip(upper=HOLIDAY_CAP_DAYS)

    return out.sort_values("date").reset_index(drop=True)


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the weekly inventory table.

    Parses ``order_week`` as datetime, normalizes ``product_category``, and
    verifies the ``ordered = sold + wasted`` identity.

    Args:
        df: Raw dataframe loaded from ``fern_inventory.csv``.

    Returns:
        Cleaned copy with typed columns.
    """
    out = df.drop_duplicates().copy()
    out["order_week"] = pd.to_datetime(out["order_week"], errors="raise")
    out["product_category"] = out["product_category"].astype("string").str.strip().str.lower()

    identity_bad = (out["units_ordered"] - out["units_sold"] - out["units_wasted"]).ne(0)
    if identity_bad.any():
        logger.warning("inventory: %d rows violate ordered = sold + wasted", int(identity_bad.sum()))

    return out.sort_values(["order_week", "product_category"]).reset_index(drop=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the customer reviews table.

    Parses ``review_date`` as datetime, lowercases ``review_text`` into a new
    ``review_text_clean`` column (original preserved), and normalizes the
    ``occasion_mentioned`` and ``platform`` columns.

    Args:
        df: Raw dataframe loaded from ``fern_reviews.csv``.

    Returns:
        Cleaned copy with typed columns.
    """
    out = df.drop_duplicates(subset=["review_id"]).copy()
    out["review_date"] = pd.to_datetime(out["review_date"], errors="raise")
    out["platform"] = out["platform"].astype("string").str.strip().str.lower()
    out["occasion_mentioned"] = out["occasion_mentioned"].astype("string").str.strip().str.lower()
    out["review_text"] = out["review_text"].astype("string").str.strip()
    out["review_text_clean"] = out["review_text"].str.lower()

    return out.sort_values("review_date").reset_index(drop=True)
