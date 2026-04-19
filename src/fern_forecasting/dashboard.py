"""Helpers for the Streamlit management dashboard.

Provides cached data loaders, an "as-of" demo clock, per-category demand
forecasts that honor that clock, a reorder simulator, and a rule-based
alert generator used by the exception-driven home page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

FEATURE_COLS: list[str] = [
    "week_of_year",
    "is_holiday_week",
    "min_days_to_holiday",
    "is_university_event_week",
    "avg_temp_f",
    "total_precipitation_inches",
    "quantity_sold_lag1",
    "quantity_sold_roll4_mean",
]
TARGET = "quantity_sold"
DEFAULT_AS_OF = pd.Timestamp("2024-10-07")
FORECAST_HORIZON_WEEKS = 8


def load_weekly_panel() -> pd.DataFrame:
    """Load the weekly panel and normalize dtypes for modeling."""
    df = pd.read_parquet(PROCESSED_DIR / "weekly_panel.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["is_holiday_week"] = df["is_holiday_week"].astype(int)
    df["is_university_event_week"] = df["is_university_event_week"].astype(int)
    df["min_days_to_holiday"] = pd.to_numeric(df["min_days_to_holiday"], errors="coerce")
    # unit_cost / unit_price can be NaN on no-activity weeks; carry the
    # category's most recent known value forward so downstream arithmetic is safe.
    df = df.sort_values(["product_category", "week_start"])
    for col in ("unit_cost", "unit_price"):
        df[col] = df.groupby("product_category")[col].ffill().bfill()
    return df.reset_index(drop=True)


def load_reviews() -> pd.DataFrame:
    """Load the cleaned reviews parquet."""
    df = pd.read_parquet(PROCESSED_DIR / "reviews_clean.parquet")
    df["review_date"] = pd.to_datetime(df["review_date"])
    return df


def load_orders() -> pd.DataFrame:
    """Load the cleaned orders parquet."""
    df = pd.read_parquet(PROCESSED_DIR / "orders_clean.parquet")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def list_categories(panel: pd.DataFrame) -> list[str]:
    """Return sorted product categories present in the panel."""
    return sorted(panel["product_category"].dropna().unique().tolist())


def snap_to_monday(ts: pd.Timestamp | pd.DatetimeIndex) -> pd.Timestamp:
    """Snap a timestamp back to the Monday of its week."""
    ts = pd.Timestamp(ts)
    return ts - pd.Timedelta(days=ts.weekday())


def as_of_bounds(panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Allowable range for the demo clock.

    Needs ~26 weeks of history behind for model training, and room ahead
    for the forecast horizon.
    """
    mondays = panel["week_start"].drop_duplicates().sort_values()
    lower = snap_to_monday(mondays.iloc[0] + pd.Timedelta(weeks=26))
    upper = snap_to_monday(mondays.iloc[-1] - pd.Timedelta(weeks=FORECAST_HORIZON_WEEKS))
    return lower, upper


def _prepare_category(panel: pd.DataFrame, category: str) -> pd.DataFrame:
    sub = panel[panel["product_category"] == category].sort_values("week_start").copy()
    return sub.dropna(subset=FEATURE_COLS + [TARGET])


def fit_forecast(
    panel: pd.DataFrame,
    category: str,
    as_of: pd.Timestamp,
    horizon: int = FORECAST_HORIZON_WEEKS,
) -> dict:
    """Train a per-category GBM on history through ``as_of`` and score ahead.

    The returned dict has:
      - ``history``: rows with ``week_start <= as_of``
      - ``forecast``: rows with ``as_of < week_start <= as_of + horizon``,
        with a ``forecast`` column (plus ``quantity_sold`` if known — used
        only for retrospective diagnostics, never surfaced as 'actual' in
        the forward-looking UI).
      - ``holdout_r2`` / ``holdout_mae``: scored on the 12 weeks immediately
        preceding ``as_of`` (a rolling evaluation window, manager-friendlier
        than a fixed 2024 test split).

    Returns ``{"error": ...}`` if there isn't enough history.
    """
    sub = _prepare_category(panel, category)
    history = sub[sub["week_start"] <= as_of]
    forecast_range = sub[
        (sub["week_start"] > as_of)
        & (sub["week_start"] <= as_of + pd.Timedelta(weeks=horizon))
    ]

    if len(history) < 26 or forecast_range.empty:
        return {"error": f"Not enough data for {category} as of {as_of.date()}"}

    holdout_start = as_of - pd.Timedelta(weeks=12)
    train = history[history["week_start"] < holdout_start]
    holdout = history[history["week_start"] >= holdout_start]

    if len(train) < 20 or holdout.empty:
        return {"error": f"Not enough training data for {category}"}

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(train[FEATURE_COLS], train[TARGET])
    holdout_pred = np.maximum(model.predict(holdout[FEATURE_COLS]), 0.0)

    # Refit on all history up to as_of for the forward forecast.
    full_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    full_model.fit(history[FEATURE_COLS], history[TARGET])
    forecast_vals = np.maximum(full_model.predict(forecast_range[FEATURE_COLS]), 0.0)

    return {
        "model": full_model,
        "history": history,
        "holdout": holdout.assign(forecast=holdout_pred),
        "forecast": forecast_range.assign(forecast=forecast_vals),
        "holdout_mae": float(mean_absolute_error(holdout[TARGET], holdout_pred)),
        "holdout_r2": float(r2_score(holdout[TARGET], holdout_pred)),
    }


def fit_all_forecasts(panel: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, dict]:
    """Fit one forecast per category, skipping categories with insufficient data."""
    results: dict[str, dict] = {}
    for cat in list_categories(panel):
        res = fit_forecast(panel, cat, as_of)
        if "error" not in res:
            results[cat] = res
    return results


def simulate_reorder(frame: pd.DataFrame, buffer_pct: float) -> pd.DataFrame:
    """Apply a ``forecast * (1 + buffer)`` reorder rule and compute waste.

    Works on any frame containing ``forecast``, ``quantity_sold``,
    ``unit_cost``, and ``units_wasted`` columns.
    """
    df = frame.copy()
    multiplier = 1.0 + buffer_pct / 100.0
    df["recommended_order"] = np.ceil(df["forecast"] * multiplier).clip(lower=0)
    df["simulated_unsold"] = (df["recommended_order"] - df["quantity_sold"]).clip(lower=0)
    unit_cost = df["unit_cost"].fillna(0.0)
    df["simulated_waste_cost"] = df["simulated_unsold"] * unit_cost
    df["actual_waste_cost"] = df["units_wasted"].fillna(0.0) * unit_cost
    return df


# ---------------------------------------------------------------------------
# Alert generation — the "manage by exception" layer
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """A single flag surfaced on the manager's home screen.

    Attributes:
        severity: "critical", "warning", "info", or "ok". Maps to the
            Streamlit alert widgets (error / warning / info / success).
        title: short headline shown in bold.
        body: supporting sentence(s).
        page: optional page file name to deep-link to.
    """

    severity: str
    title: str
    body: str
    page: str | None = None


def _recent_window(df: pd.DataFrame, date_col: str, as_of: pd.Timestamp, weeks: int):
    start = as_of - pd.Timedelta(weeks=weeks)
    return df[(df[date_col] > start) & (df[date_col] <= as_of)]


def generate_alerts(
    panel: pd.DataFrame,
    forecasts: dict[str, dict],
    reviews: pd.DataFrame,
    as_of: pd.Timestamp,
    buffer_pct: float = 5.0,
) -> list[Alert]:
    """Produce a prioritized list of alerts for the ``as_of`` moment."""
    alerts: list[Alert] = []

    # --- Demand spikes ahead (top 2 spikes in the next 4 weeks) ------------
    spike_candidates: list[tuple[float, str, pd.Series, float, int]] = []
    for cat, res in forecasts.items():
        hist = res["history"]
        baseline = hist.tail(4)[TARGET].mean() if len(hist) >= 4 else np.nan
        next4 = res["forecast"].head(4)
        if np.isnan(baseline) or baseline <= 0 or next4.empty:
            continue
        peak_row = next4.loc[next4["forecast"].idxmax()]
        pct = (peak_row["forecast"] / baseline - 1.0) * 100.0
        if pct >= 30:
            weeks_out = int((peak_row["week_start"] - as_of).days / 7)
            spike_candidates.append((pct, cat, peak_row, baseline, weeks_out))
    for pct, cat, peak_row, baseline, weeks_out in sorted(spike_candidates, reverse=True)[:2]:
        alerts.append(
            Alert(
                severity="warning",
                title=f"{cat.title()}: demand forecast +{pct:.0f}% in {weeks_out} wk",
                body=(
                    f"Forecast {peak_row['forecast']:.0f} units for week of "
                    f"{peak_row['week_start'].date()} vs. 4-week avg "
                    f"{baseline:.0f}. Consider front-loading the order."
                ),
                page="Demand_Outlook",
            )
        )

    # --- Recent excess waste (top 2 categories by waste $ last 2 weeks) ----
    hist_all = panel[panel["week_start"] <= as_of]
    last2 = _recent_window(hist_all, "week_start", as_of, 2).copy()
    last2["waste_cost"] = last2["units_wasted"].fillna(0) * last2["unit_cost"].fillna(0)
    waste_by_cat = last2.groupby("product_category")["waste_cost"].sum().sort_values(ascending=False)
    for cat, cost in waste_by_cat.head(2).items():
        if cost >= 50:
            alerts.append(
                Alert(
                    severity="warning",
                    title=f"{cat.title()}: ${cost:.0f} wasted in last 2 weeks",
                    body="Buffer may be too generous — run the simulation and compare.",
                    page="How_We_Did",
                )
            )

    # --- Potential stockouts (tight criterion: both recent weeks fully sold) ---
    last2_nonzero = last2[last2["units_ordered"].fillna(0) >= 5].copy()
    last2_nonzero["sellthrough"] = (
        last2_nonzero["units_sold"].fillna(0) / last2_nonzero["units_ordered"]
    )
    # Flag only if every recent order in this category was fully sold through,
    # AND the category has at least 2 weeks of history in the window.
    sellthrough_by_cat = (
        last2_nonzero.groupby("product_category")
        .agg(n_weeks=("sellthrough", "size"), min_st=("sellthrough", "min"))
    )
    sold_out = sellthrough_by_cat[
        (sellthrough_by_cat["n_weeks"] >= 2) & (sellthrough_by_cat["min_st"] >= 1.0)
    ].index.tolist()
    for cat in sold_out[:2]:
        alerts.append(
            Alert(
                severity="critical",
                title=f"{cat.title()}: fully sold through — possible lost sales",
                body=(
                    "Every unit ordered in the last 2 weeks was sold. Bump the safety "
                    "buffer or review whether the forecast is under-reading demand."
                ),
                page="Order_Sheet",
            )
        )

    # --- Sentiment drift (last 30 days vs prior 90) ------------------------
    r30 = _recent_window(reviews, "review_date", as_of, 4)  # ~30 days
    r_prior = reviews[
        (reviews["review_date"] <= as_of - pd.Timedelta(weeks=4))
        & (reviews["review_date"] > as_of - pd.Timedelta(weeks=17))
    ]
    if not r30.empty and not r_prior.empty and "sentiment" in reviews.columns:
        recent_sent = r30["sentiment"].mean()
        prior_sent = r_prior["sentiment"].mean()
        if recent_sent < prior_sent - 0.10:
            alerts.append(
                Alert(
                    severity="warning",
                    title=f"Sentiment dipped {prior_sent - recent_sent:.2f} vs prior 90 days",
                    body=(
                        f"Last 30 days avg sentiment {recent_sent:+.2f} "
                        f"(prior 90 days {prior_sent:+.2f}). Review recent feedback."
                    ),
                    page="Customer_Voice",
                )
            )

    # --- Negative reviews in last 14 days ---------------------------------
    r14_neg = reviews[
        (reviews["review_date"] > as_of - pd.Timedelta(weeks=2))
        & (reviews["review_date"] <= as_of)
        & (reviews["star_rating"] <= 2)
    ]
    if not r14_neg.empty:
        n = len(r14_neg)
        alerts.append(
            Alert(
                severity="critical",
                title=f"{n} low-star review{'s' if n != 1 else ''} in last 2 weeks",
                body="Surface and respond before they influence new customers.",
                page="Customer_Voice",
            )
        )

    # --- Upcoming holiday within 4 weeks ----------------------------------
    next4_weeks = panel[
        (panel["week_start"] > as_of)
        & (panel["week_start"] <= as_of + pd.Timedelta(weeks=4))
    ]
    holidays = next4_weeks[next4_weeks["is_holiday_week"] == 1].drop_duplicates("week_start")
    if not holidays.empty:
        row = holidays.sort_values("week_start").iloc[0]
        name = row.get("holiday_names") or "a holiday"
        if pd.isna(name):
            name = "a holiday"
        weeks_out = int((row["week_start"] - as_of).days / 7)
        alerts.append(
            Alert(
                severity="info",
                title=f"Holiday ahead: {name} in {weeks_out} wk",
                body=(
                    f"Week of {row['week_start'].date()} is flagged as a holiday week. "
                    "Review forecasts and buffer per category."
                ),
                page="Demand_Outlook",
            )
        )

    # --- Always: this-week order ready ------------------------------------
    current_week_rows = []
    target_week = snap_to_monday(as_of) + pd.Timedelta(weeks=1)
    for cat, res in forecasts.items():
        hit = res["forecast"][res["forecast"]["week_start"] == target_week]
        if hit.empty:
            continue
        row = hit.iloc[0]
        units = int(np.ceil(row["forecast"] * (1 + buffer_pct / 100.0)))
        unit_cost = row.get("unit_cost", 0.0)
        unit_cost = float(unit_cost) if pd.notna(unit_cost) else 0.0
        current_week_rows.append((cat, units, units * unit_cost))
    if current_week_rows:
        total_units = sum(r[1] for r in current_week_rows)
        total_cost = sum(r[2] for r in current_week_rows)
        alerts.append(
            Alert(
                severity="info",
                title=f"This week's order ready — {total_units} units, ${total_cost:,.0f}",
                body=f"Recommended buy for week of {target_week.date()}. Open the order sheet to review.",
                page="Order_Sheet",
            )
        )

    if not alerts:
        alerts.append(
            Alert(
                severity="ok",
                title="All systems normal",
                body="No exceptions detected for this week.",
            )
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2, "ok": 3}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 99))
    return alerts


# ---------------------------------------------------------------------------
# UI glue — kept here so every page uses the same as-of selector
# ---------------------------------------------------------------------------


def render_as_of_sidebar(panel: pd.DataFrame, key: str = "as_of") -> pd.Timestamp:
    """Render the shared 'as of' date picker in the sidebar.

    Snaps the user's pick to the Monday of that week. Stores and reads
    from ``st.session_state[key]`` so the date persists across pages.
    """
    import streamlit as st  # local import so the module stays usable outside Streamlit

    lower, upper = as_of_bounds(panel)
    if key not in st.session_state:
        st.session_state[key] = max(min(DEFAULT_AS_OF, upper), lower).date()

    picked = st.sidebar.date_input(
        "Dashboard 'as of'",
        value=st.session_state[key],
        min_value=lower.date(),
        max_value=upper.date(),
        help=(
            "Pretend today is this date. History = before, forecasts = after. "
            "Snaps to the Monday of the chosen week."
        ),
    )
    as_of = snap_to_monday(pd.Timestamp(picked))
    st.session_state[key] = as_of.date()

    st.sidebar.caption(f"Treating **{as_of:%a, %b %d, %Y}** as 'today'.")
    return as_of
