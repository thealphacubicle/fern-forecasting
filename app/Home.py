"""Fern Forecasting — manager-facing home screen.

Designed to be scannable in under 10 seconds: the shop owner lands here,
sees any exceptions at the top, and drills into the relevant page only if
something needs their attention.

Run from the repo root::

    uv run streamlit run app/Home.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fern_forecasting.dashboard import (
    fit_all_forecasts,
    generate_alerts,
    load_reviews_scored,
    load_weekly_panel,
    render_as_of_sidebar,
    snap_to_monday,
)

st.set_page_config(page_title="Fern Forecasting", layout="wide")


@st.cache_data
def _panel() -> pd.DataFrame:
    return load_weekly_panel()


@st.cache_data
def _reviews_scored() -> pd.DataFrame:
    return load_reviews_scored()


@st.cache_resource
def _forecasts(_panel_df: pd.DataFrame, as_of_iso: str) -> dict:
    return fit_all_forecasts(_panel_df, pd.Timestamp(as_of_iso))


panel = _panel()
reviews = _reviews_scored()

as_of = render_as_of_sidebar(panel)
buffer_pct = st.sidebar.slider(
    "Safety buffer (%)",
    min_value=0,
    max_value=30,
    value=int(st.session_state.get("buffer_pct", 5)),
    step=1,
    help="Applied to forecasts to produce the recommended weekly order.",
)
st.session_state["buffer_pct"] = buffer_pct

forecasts = _forecasts(panel, as_of.isoformat())

# ---- Header ---------------------------------------------------------------
title_col, badge_col = st.columns([4, 1])
with title_col:
    st.title("Fern — Morning Briefing")
with badge_col:
    st.markdown(
        f"<div style='text-align:right; margin-top:18px;'>"
        f"<span style='background:#eef; padding:6px 10px; border-radius:6px; "
        f"font-size:0.9em;'>as of <b>{as_of:%a, %b %d, %Y}</b></span></div>",
        unsafe_allow_html=True,
    )

# ---- Last-week KPIs -------------------------------------------------------
last_week = snap_to_monday(as_of) - pd.Timedelta(weeks=0)  # as_of IS a Monday
prior_week = last_week - pd.Timedelta(weeks=1)

lw_rows = panel[panel["week_start"] == last_week]
pw_rows = panel[panel["week_start"] == prior_week]

lw_rev = float(lw_rows["revenue"].sum())
pw_rev = float(pw_rows["revenue"].sum())
lw_units = int(lw_rows["units_sold"].fillna(0).sum())
lw_waste_cost = float((lw_rows["units_wasted"].fillna(0) * lw_rows["unit_cost"].fillna(0)).sum())
top_seller = (
    lw_rows.sort_values("quantity_sold", ascending=False).iloc[0]["product_category"]
    if not lw_rows.empty
    else "—"
)

st.subheader(f"Last week at a glance — week of {last_week:%b %d}")
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Revenue",
    f"${lw_rev:,.0f}",
    delta=f"{(lw_rev - pw_rev):+,.0f} vs prior week" if pw_rev else None,
)
k2.metric("Units sold", f"{lw_units:,}")
k3.metric("Waste cost", f"${lw_waste_cost:,.0f}")
k4.metric("Top seller", top_seller.title())

st.divider()

# ---- Alerts feed ----------------------------------------------------------
st.subheader("What needs your attention")
st.caption(
    "Automated exceptions across demand, inventory, and customer sentiment. "
    "Click through to the relevant page for detail."
)

alerts = generate_alerts(panel, forecasts, reviews, as_of, buffer_pct=buffer_pct)
severity_renderer = {
    "critical": st.error,
    "warning": st.warning,
    "info": st.info,
    "ok": st.success,
}
for alert in alerts:
    render = severity_renderer.get(alert.severity, st.info)
    footer = f"\n\n→ See **{alert.page.replace('_', ' ')}**" if alert.page else ""
    render(f"**{alert.title}**\n\n{alert.body}{footer}")

st.divider()
st.caption(
    "Use the left-hand pages to dig in: Demand Outlook, Order Sheet, How We Did, "
    "Customer Voice. The 'as of' date in the sidebar controls what the dashboard "
    "treats as today."
)
