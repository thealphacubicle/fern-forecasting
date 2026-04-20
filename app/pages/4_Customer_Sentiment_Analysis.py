"""Customer Sentiment Analysis: recent reviews, flagged exceptions, occasion satisfaction."""

from __future__ import annotations

import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from fern_forecasting.dashboard import (
    load_reviews,
    load_weekly_panel,
    render_as_of_sidebar,
)

st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")


@st.cache_data
def _panel() -> pd.DataFrame:
    return load_weekly_panel()


@st.cache_resource
def _analyzer() -> SentimentIntensityAnalyzer:
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_data
def _reviews() -> pd.DataFrame:
    df = load_reviews()
    analyzer = _analyzer()
    df["sentiment"] = df["review_text_clean"].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    return df


panel = _panel()
as_of = render_as_of_sidebar(panel)
reviews = _reviews()

visible = reviews[reviews["review_date"] <= as_of].sort_values("review_date", ascending=False)

st.title("Customer Sentiment Analysis")
st.caption(f"Reviews and sentiment as of {as_of:%a, %b %d, %Y}.")

# ---- Exception band -------------------------------------------------------
r14 = visible[visible["review_date"] > as_of - pd.Timedelta(weeks=2)]
r14_neg = r14[r14["star_rating"] <= 2]
r30 = visible[visible["review_date"] > as_of - pd.Timedelta(weeks=4)]
r_prior = visible[
    (visible["review_date"] <= as_of - pd.Timedelta(weeks=4))
    & (visible["review_date"] > as_of - pd.Timedelta(weeks=17))
]

flagged = False
if not r14_neg.empty:
    flagged = True
    st.error(
        f"**{len(r14_neg)} low-star review{'s' if len(r14_neg) != 1 else ''} in the "
        f"last 2 weeks.** Scroll to the flagged section below to respond."
    )
if not r30.empty and not r_prior.empty:
    recent_sent = r30["sentiment"].mean()
    prior_sent = r_prior["sentiment"].mean()
    if recent_sent < prior_sent - 0.10:
        flagged = True
        st.warning(
            f"Sentiment dipped **{prior_sent - recent_sent:.2f}** vs. the prior 90 days "
            f"(recent {recent_sent:+.2f}, prior {prior_sent:+.2f})."
        )
if not flagged:
    st.success("No customer-voice exceptions in the last 30 days.")

# ---- KPIs ----------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Reviews (all-time)", f"{len(visible):,}")
k2.metric("Avg rating", f"{visible['star_rating'].mean():.2f} / 5")
k3.metric("Avg sentiment", f"{visible['sentiment'].mean():+.2f}")
k4.metric("Reviews in last 30d", f"{len(r30):,}")

# ---- Unresolved low-star reviews (surfaced first) ------------------------
st.subheader("Flagged reviews — last 14 days, ≤ 2★")
if r14_neg.empty:
    st.caption("None — clean week.")
else:
    for _, row in r14_neg.iterrows():
        with st.container(border=True):
            st.markdown(
                f"**{int(row['star_rating'])}★ — {row['platform'].title()}** · "
                f"{row['review_date']:%b %d, %Y} · occasion: _{row['occasion_mentioned']}_"
            )
            st.write(row["review_text"])

# ---- Recent review stream ------------------------------------------------
st.subheader("Recent reviews")
recent_window_weeks = st.slider("Window (weeks)", 2, 26, 8, key="cv_window")
stream = visible[visible["review_date"] > as_of - pd.Timedelta(weeks=recent_window_weeks)]
show_cols = ["review_date", "platform", "star_rating", "occasion_mentioned", "sentiment", "review_text"]
st.dataframe(
    stream[show_cols].reset_index(drop=True),
    width="stretch",
    hide_index=True,
)

# ---- Occasion leaderboard ------------------------------------------------
st.subheader("Occasion satisfaction — leaderboard")
st.caption("Which occasions delight customers most, which leave them cold.")
by_occasion = (
    visible.groupby("occasion_mentioned")
    .agg(
        reviews=("review_id", "count"),
        avg_rating=("star_rating", "mean"),
        avg_sentiment=("sentiment", "mean"),
    )
    .sort_values("avg_sentiment", ascending=False)
    .round(3)
)
st.bar_chart(by_occasion["avg_sentiment"], height=280)
st.dataframe(by_occasion, width="stretch")

# ---- Sentiment trend -----------------------------------------------------
st.subheader("Sentiment trend (monthly)")
trend = visible.set_index("review_date")["sentiment"].resample("MS").mean().dropna()
st.line_chart(trend, height=260)
