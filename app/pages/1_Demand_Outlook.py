"""Demand Outlook — per-category history and 8-week forward forecast."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fern_forecasting.dashboard import (
    FEATURE_COLS,
    TARGET,
    fit_all_forecasts,
    list_categories,
    load_weekly_panel,
    render_as_of_sidebar,
)

st.set_page_config(page_title="Demand Outlook", layout="wide")


@st.cache_data
def _panel() -> pd.DataFrame:
    return load_weekly_panel()


@st.cache_resource
def _forecasts(_panel_df: pd.DataFrame, as_of_iso: str) -> dict:
    return fit_all_forecasts(_panel_df, pd.Timestamp(as_of_iso))


panel = _panel()
as_of = render_as_of_sidebar(panel)
forecasts = _forecasts(panel, as_of.isoformat())

categories = [c for c in list_categories(panel) if c in forecasts]
if not categories:
    st.error("Not enough history to forecast as of this date.")
    st.stop()

category = st.sidebar.selectbox("Product category", categories, index=0)
res = forecasts[category]
history = res["history"]
forecast = res["forecast"]

# ---- Header ---------------------------------------------------------------
st.title(f"{category.title()} — Demand Outlook")
st.caption(
    f"Weekly history through {as_of:%b %d, %Y} and forecast for the next "
    f"{len(forecast)} weeks."
)

# ---- Exception band: what's notable about this category -------------------
exceptions: list[tuple[str, str]] = []
recent = history.tail(4)
baseline = recent[TARGET].mean() if not recent.empty else 0
if not forecast.empty and baseline > 0:
    peak = forecast.loc[forecast["forecast"].idxmax()]
    pct = (peak["forecast"] / baseline - 1.0) * 100.0
    if pct >= 25:
        exceptions.append(
            (
                "warning",
                f"Forecast peak of **{peak['forecast']:.0f} units** for week of "
                f"{peak['week_start']:%b %d} — +{pct:.0f}% vs. last 4-week average "
                f"({baseline:.0f}).",
            )
        )
    if pct <= -25:
        exceptions.append(
            (
                "info",
                f"Demand expected to cool to **{peak['forecast']:.0f} units** — "
                f"{pct:.0f}% vs. last 4-week average.",
            )
        )

upcoming_holidays = forecast[forecast["is_holiday_week"] == 1]
if not upcoming_holidays.empty:
    hits = ", ".join(
        f"{r['week_start']:%b %d} ({r.get('holiday_names') or 'holiday'})"
        for _, r in upcoming_holidays.iterrows()
    )
    exceptions.append(("info", f"Upcoming holiday weeks in the forecast window: {hits}."))

if exceptions:
    for sev, msg in exceptions:
        (st.warning if sev == "warning" else st.info)(msg)
else:
    st.success("Nothing unusual in the outlook — steady trend.")

# ---- Timeline chart -------------------------------------------------------
window_start = as_of - pd.Timedelta(weeks=13)
hist_window = history[history["week_start"] >= window_start][["week_start", TARGET]]
hist_window = hist_window.rename(columns={TARGET: "Actual"}).set_index("week_start")
fore_window = forecast[["week_start", "forecast"]].rename(columns={"forecast": "Forecast"})
fore_window = fore_window.set_index("week_start")
combined = pd.concat([hist_window, fore_window], axis=1).sort_index()

st.subheader("Weekly demand — 13 weeks past + forecast ahead")
st.line_chart(combined, height=360)

# ---- Year-over-year sanity check -----------------------------------------
ya = as_of - pd.Timedelta(weeks=52)
ya_rows = history[
    (history["week_start"] >= ya - pd.Timedelta(weeks=4))
    & (history["week_start"] <= ya + pd.Timedelta(weeks=4))
]
this_rows = history[history["week_start"] >= as_of - pd.Timedelta(weeks=4)]
if not ya_rows.empty and not this_rows.empty:
    st.subheader("Same period last year vs. now")
    ya_total = float(ya_rows[TARGET].sum())
    now_total = float(this_rows[TARGET].sum())
    change = now_total - ya_total
    pct = (change / ya_total * 100.0) if ya_total else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{ya:%b %Y} ± 4 wk", f"{ya_total:,.0f} units")
    c2.metric("Last 4 weeks", f"{now_total:,.0f} units", f"{change:+,.0f} ({pct:+.1f}%)")
    c3.metric(
        "Next 4 weeks (forecast)",
        f"{forecast.head(4)['forecast'].sum():,.0f} units",
    )

# ---- Detail tables --------------------------------------------------------
with st.expander("Forecast detail — next 8 weeks"):
    out = forecast[["week_start", "forecast", "is_holiday_week", "min_days_to_holiday"]].copy()
    out["forecast"] = out["forecast"].round(1)
    out = out.rename(
        columns={
            "week_start": "Week",
            "forecast": "Forecast (units)",
            "is_holiday_week": "Holiday week?",
            "min_days_to_holiday": "Days to holiday",
        }
    )
    st.dataframe(out, width="stretch", hide_index=True)

with st.expander("Model diagnostics (for the analyst)"):
    d1, d2, d3 = st.columns(3)
    d1.metric("Rolling 12-wk R²", f"{res['holdout_r2']:+.2f}")
    d2.metric("Rolling 12-wk MAE", f"{res['holdout_mae']:.1f} units")
    d3.metric("Training weeks", f"{len(res['history'])}")
    st.caption(
        "Scored on the 12 weeks immediately before the 'as of' date — a rolling "
        "evaluation that tracks how the model does on fresh data."
    )
    importances = pd.Series(
        res["model"].feature_importances_, index=FEATURE_COLS, name="importance"
    ).sort_values(ascending=False)
    st.bar_chart(importances, height=260)
