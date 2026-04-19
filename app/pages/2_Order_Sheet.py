"""Order Sheet — this week's recommended order, in a supplier-ready layout."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fern_forecasting.dashboard import (
    fit_all_forecasts,
    load_weekly_panel,
    render_as_of_sidebar,
    simulate_reorder,
    snap_to_monday,
)

st.set_page_config(page_title="Order Sheet", layout="wide")


@st.cache_data
def _panel() -> pd.DataFrame:
    return load_weekly_panel()


@st.cache_resource
def _forecasts(_panel_df: pd.DataFrame, as_of_iso: str) -> dict:
    return fit_all_forecasts(_panel_df, pd.Timestamp(as_of_iso))


panel = _panel()
as_of = render_as_of_sidebar(panel)
forecasts = _forecasts(panel, as_of.isoformat())

buffer_pct = st.sidebar.slider(
    "Safety buffer (%)",
    min_value=0,
    max_value=30,
    value=int(st.session_state.get("buffer_pct", 5)),
    step=1,
)
st.session_state["buffer_pct"] = buffer_pct

# ---- Week picker: default to next Monday ---------------------------------
available_weeks = sorted(
    {pd.Timestamp(w) for r in forecasts.values() for w in r["forecast"]["week_start"].unique()}
)
if not available_weeks:
    st.error("No forecast weeks available.")
    st.stop()

default_week = snap_to_monday(as_of) + pd.Timedelta(weeks=1)
if default_week not in available_weeks:
    default_week = available_weeks[0]

selected = st.sidebar.select_slider(
    "Order for week of",
    options=[w.date() for w in available_weeks],
    value=default_week.date(),
)
target_week = pd.Timestamp(selected)
weeks_out = int((target_week - as_of).days / 7)

# ---- Build the order sheet -----------------------------------------------
rows: list[dict] = []
for cat, res in forecasts.items():
    hit = res["forecast"][res["forecast"]["week_start"] == target_week]
    if hit.empty:
        continue
    sim = simulate_reorder(hit, buffer_pct).iloc[0]

    # "Same week last year" context so the manager can sanity-check.
    ya = target_week - pd.Timedelta(weeks=52)
    ya_hit = res["history"][res["history"]["week_start"] == ya]
    ya_sold = int(ya_hit["quantity_sold"].iloc[0]) if not ya_hit.empty else None

    rows.append(
        {
            "Category": cat.title(),
            "Forecast (units)": round(float(sim["forecast"]), 1),
            "Buffer": f"{buffer_pct}%",
            "Order (units)": int(sim["recommended_order"]),
            "Unit cost": round(float(sim.get("unit_cost") or 0), 2),
            "Line total": round(
                float(sim["recommended_order"]) * float(sim.get("unit_cost") or 0), 2
            ),
            "Same wk last yr": ya_sold if ya_sold is not None else "—",
            "Holiday wk?": "Yes" if sim.get("is_holiday_week") else "No",
        }
    )

order_df = pd.DataFrame(rows)
if order_df.empty:
    st.warning("No data for the selected week.")
    st.stop()

# ---- Header ---------------------------------------------------------------
st.title("Order Sheet")
st.caption(
    f"Recommended buy for **week of {target_week:%a, %b %d, %Y}** "
    f"({weeks_out} week{'s' if weeks_out != 1 else ''} out). "
    f"Lead time is typically 3 days — submit by Friday for Monday delivery."
)

# ---- Exception band -------------------------------------------------------
holiday_cats = order_df[order_df["Holiday wk?"] == "Yes"]
if not holiday_cats.empty:
    st.warning(
        f"Holiday week — demand typically surges. Consider raising the buffer above "
        f"{buffer_pct}% or placing a second mid-week order."
    )

spike_cats: list[str] = []
for _, row in order_df.iterrows():
    ya_val = row["Same wk last yr"]
    if isinstance(ya_val, (int, float)) and ya_val > 0:
        if row["Order (units)"] >= ya_val * 1.25:
            spike_cats.append(row["Category"])
if spike_cats:
    st.info(
        "Bigger order than same week last year for: " + ", ".join(spike_cats) + "."
    )

# ---- Summary KPIs ---------------------------------------------------------
total_cost = float(order_df["Line total"].sum())
total_units = int(order_df["Order (units)"].sum())
total_forecast = float(order_df["Forecast (units)"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Order total", f"${total_cost:,.2f}")
c2.metric("Total units", f"{total_units:,}")
c3.metric("Forecast units", f"{total_forecast:,.1f}")

# ---- Order table ---------------------------------------------------------
st.subheader("Per-category order")
st.dataframe(order_df, width="stretch", hide_index=True)

# ---- Actions (mock) ------------------------------------------------------
action_col1, action_col2, _ = st.columns([1, 1, 4])
if action_col1.button("Send to supplier", type="primary"):
    st.toast("Order queued for supplier (demo).", icon="✉")
if action_col2.download_button(
    "Download CSV",
    data=order_df.to_csv(index=False).encode(),
    file_name=f"fern_order_{target_week:%Y%m%d}.csv",
    mime="text/csv",
):
    pass

# ---- Why this amount? ----------------------------------------------------
with st.expander("Why these numbers?"):
    st.markdown(
        f"""
- **Forecast** comes from the per-category Gradient Boosting model trained on weekly
  history through {as_of:%b %d, %Y}.
- **Order** = `ceil(forecast × (1 + buffer%))` — the buffer covers forecast error and
  unexpected walk-ins.
- **Same week last year** is the raw units sold in the corresponding week of
  {(target_week - pd.Timedelta(weeks=52)).year} — useful as a sanity check, especially
  for calendar-driven demand (Valentine's, Mother's Day, graduation).
- **Unit cost** is Fern's wholesale cost from the inventory table.
"""
    )
