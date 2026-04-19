"""How We Did — retrospective view of recent weeks.

Shows last week, last 4 weeks, and last 12 weeks of operational performance,
with a supporting "what if" buffer simulator for the manager to probe.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fern_forecasting.dashboard import (
    fit_all_forecasts,
    load_weekly_panel,
    render_as_of_sidebar,
    simulate_reorder,
)

st.set_page_config(page_title="How We Did", layout="wide")


@st.cache_data
def _panel() -> pd.DataFrame:
    return load_weekly_panel()


@st.cache_resource
def _forecasts(_panel_df: pd.DataFrame, as_of_iso: str) -> dict:
    return fit_all_forecasts(_panel_df, pd.Timestamp(as_of_iso))


panel = _panel()
as_of = render_as_of_sidebar(panel)
forecasts = _forecasts(panel, as_of.isoformat())

st.title("How We Did")
st.caption(f"Looking back from {as_of:%a, %b %d, %Y}.")

hist = panel[panel["week_start"] <= as_of].copy()
hist["waste_cost"] = hist["units_wasted"].fillna(0) * hist["unit_cost"].fillna(0)

# ---- Last week ------------------------------------------------------------
last_week = hist["week_start"].max()
prior_week = last_week - pd.Timedelta(weeks=1)
ly_week = last_week - pd.Timedelta(weeks=52)

lw = hist[hist["week_start"] == last_week]
pw = hist[hist["week_start"] == prior_week]
ly = hist[hist["week_start"] == ly_week]

st.subheader(f"Last week — {last_week:%b %d}")
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Revenue",
    f"${lw['revenue'].sum():,.0f}",
    delta=f"{(lw['revenue'].sum() - pw['revenue'].sum()):+,.0f} vs prior",
)
k2.metric(
    "Units sold",
    f"{int(lw['units_sold'].fillna(0).sum()):,}",
    delta=f"{int(lw['units_sold'].fillna(0).sum() - ly['units_sold'].fillna(0).sum()):+,} vs 1yr ago"
    if not ly.empty
    else None,
)
k3.metric("Waste cost", f"${lw['waste_cost'].sum():,.0f}")
if not lw.empty:
    top = lw.sort_values("quantity_sold", ascending=False).iloc[0]
    k4.metric("Top seller", top["product_category"].title(), f"{int(top['quantity_sold'])} units")

# ---- Exception band -------------------------------------------------------
flags: list[str] = []
last4 = hist[hist["week_start"] > as_of - pd.Timedelta(weeks=4)]
waste_by_cat = last4.groupby("product_category")["waste_cost"].sum()
high_waste = waste_by_cat[waste_by_cat > 40].sort_values(ascending=False)
for cat, cost in high_waste.items():
    flags.append(f"**{cat.title()}** wasted **${cost:.0f}** in the last 4 weeks")

last4_inv = last4[last4["units_ordered"].fillna(0) > 0].copy()
last4_inv["sellthrough"] = last4_inv["units_sold"].fillna(0) / last4_inv["units_ordered"]
sold_out = last4_inv[last4_inv["sellthrough"] >= 1.0]["product_category"].unique()
for cat in sold_out:
    flags.append(f"**{cat.title()}** sold out at least once in the last 4 weeks")

if flags:
    for f in flags:
        st.warning(f)
else:
    st.success("No waste or stockout exceptions in the last 4 weeks.")

# ---- Trailing trends ------------------------------------------------------
st.subheader("Last 12 weeks — revenue and waste")
trailing = hist[hist["week_start"] > as_of - pd.Timedelta(weeks=12)]
weekly = trailing.groupby("week_start")[["revenue", "waste_cost"]].sum()
weekly.columns = ["Revenue", "Waste cost"]
st.line_chart(weekly, height=320)

# ---- Last month by category ----------------------------------------------
st.subheader("Last 4 weeks by category")
by_cat = (
    trailing[trailing["week_start"] > as_of - pd.Timedelta(weeks=4)]
    .groupby("product_category")
    .agg(
        units_sold=("units_sold", lambda s: int(s.fillna(0).sum())),
        revenue=("revenue", "sum"),
        units_wasted=("units_wasted", lambda s: int(s.fillna(0).sum())),
        waste_cost=("waste_cost", "sum"),
    )
    .sort_values("revenue", ascending=False)
    .round(2)
)
by_cat.index = [c.title() for c in by_cat.index]
by_cat.index.name = "Category"
st.dataframe(by_cat, width="stretch")

# ---- Buffer what-if -------------------------------------------------------
st.divider()
st.subheader("What if we'd used the model? (last 12 weeks)")
st.caption(
    "Retrospective simulation: applies the 'forecast × (1 + buffer)' ordering rule "
    "to the last 12 weeks and compares resulting waste to what actually happened."
)

buffer_pct = st.slider(
    "Safety buffer (%)",
    min_value=0,
    max_value=30,
    value=int(st.session_state.get("buffer_pct", 5)),
    step=1,
    key="whatif_buffer",
)
st.session_state["buffer_pct"] = buffer_pct

# Use the holdout frame (which has a forecast column) from each per-category fit.
holdouts: list[pd.DataFrame] = []
for cat, res in forecasts.items():
    h = res["holdout"].copy()
    h["product_category"] = cat
    holdouts.append(h)

if holdouts:
    sim = simulate_reorder(pd.concat(holdouts, ignore_index=True), buffer_pct)
    actual_waste = sim["actual_waste_cost"].sum()
    sim_waste = sim["simulated_waste_cost"].sum()
    delta = actual_waste - sim_waste

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual waste cost", f"${actual_waste:,.2f}")
    c2.metric("Simulated waste cost", f"${sim_waste:,.2f}")
    c3.metric(
        "Delta",
        f"${delta:+,.2f}",
        delta=f"{(delta / actual_waste * 100 if actual_waste else 0):+.1f}%",
    )

    by_cat_sim = (
        sim.groupby("product_category")[["actual_waste_cost", "simulated_waste_cost"]]
        .sum()
        .rename(
            columns={
                "actual_waste_cost": "Actual",
                "simulated_waste_cost": "Simulated",
            }
        )
    )
    st.bar_chart(by_cat_sim, height=280)

    with st.expander("Caveats"):
        st.markdown(
            """
- Simulation assumes order = `ceil(forecast × (1 + buffer%))`; stockouts and
  markdowns aren't modeled, so this is an upper-bound on waste savings.
- Model is retrained on data through the 'as of' date; its 12-week holdout is
  the window used here.
"""
        )
else:
    st.info("No recent holdout data to simulate against.")
