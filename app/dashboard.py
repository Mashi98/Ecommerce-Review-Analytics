"""Streamlit dashboard: review sentiment and topics by department, class and age.

    streamlit run app/dashboard.py

A thin layer over `src/dashboard/data.py` (unit-tested) and the baseline model.
Reads the train and validation splits only; the test split stays held out.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import streamlit as st  # noqa: E402

from scripts.predict_review import influential_terms, predict_reviews  # noqa: E402
from src.dashboard.data import (  # noqa: E402
    AGE_BAND_COL,
    AGE_BAND_LABELS,
    CLASS_COL,
    DEPARTMENT_COL,
    SEGMENT_COLUMNS,
    filter_reviews,
    load_dashboard_data,
    load_topic_table,
    overview,
    sentiment_breakdown,
)
from src.models.baseline import SENTIMENT_LABELS, load_split, train_baseline  # noqa: E402

SENTIMENT_COLORS = {"negative": "#c2410c", "neutral": "#a1a1aa", "positive": "#0f766e"}
SENTIMENT_SCALE = alt.Scale(domain=SENTIMENT_LABELS, range=[SENTIMENT_COLORS[s] for s in SENTIMENT_LABELS])

st.set_page_config(page_title="Review Analytics", layout="wide")


@st.cache_data
def reviews():
    return load_dashboard_data()


@st.cache_data
def topics():
    return load_topic_table()


@st.cache_resource(show_spinner="Training the baseline on the training split...")
def baseline_model():
    return train_baseline(load_split("train"), model="logreg", imbalance="class_weight")


def sidebar_filters(df):
    st.sidebar.header("Filters")
    departments = st.sidebar.multiselect("Department", sorted(df[DEPARTMENT_COL].unique()))
    class_pool = filter_reviews(df, departments=departments)
    classes = st.sidebar.multiselect("Class", sorted(class_pool[CLASS_COL].unique()))
    age_bands = st.sidebar.multiselect("Age band", AGE_BAND_LABELS)
    st.sidebar.caption(
        "Train + validation splits (test split held out). "
        "Sentiment comes from the star rating: 1-2 negative, 3 neutral, 4-5 positive."
    )
    return filter_reviews(df, departments=departments, classes=classes, age_bands=age_bands), departments


def headline(df):
    stats = overview(df)
    cols = st.columns(5)
    cols[0].metric("Reviews", f"{stats['reviews']:,}")
    cols[1].metric("Negative", f"{stats['shares']['negative']:.1%}")
    cols[2].metric("Neutral", f"{stats['shares']['neutral']:.1%}")
    cols[3].metric("Recommend", f"{stats['recommend_rate']:.1%}" if stats["reviews"] else "n/a")
    cols[4].metric("Mean rating", f"{stats['mean_rating']:.2f}" if stats["reviews"] else "n/a")


def sentiment_chart(table, segment_label):
    height = max(120, 28 * table["segment"].nunique())
    return (
        alt.Chart(table)
        .mark_bar()
        .encode(
            y=alt.Y("segment:N", sort=None, title=segment_label),
            x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of reviews"),
            color=alt.Color("sentiment:N", scale=SENTIMENT_SCALE, sort=SENTIMENT_LABELS, title=None),
            order=alt.Order("sentiment_order:Q"),
            tooltip=[
                alt.Tooltip("segment:N", title=segment_label),
                "sentiment:N",
                alt.Tooltip("reviews:Q", format=","),
                alt.Tooltip("share:Q", format=".1%"),
                alt.Tooltip("segment_total:Q", title="Segment reviews", format=","),
            ],
        )
        .transform_calculate(sentiment_order=f"indexof({SENTIMENT_LABELS!r}, datum.sentiment)")
        .properties(height=height)
    )


def sentiment_tab(df):
    left, right = st.columns([1, 1])
    segment_label = left.radio("Break down by", list(SEGMENT_COLUMNS), horizontal=True)
    min_reviews = right.slider("Hide segments with fewer reviews than", 1, 200, 30)
    table = sentiment_breakdown(df, SEGMENT_COLUMNS[segment_label], min_reviews=min_reviews)
    if table.empty:
        st.info("No segment has enough reviews under the current filters.")
        return
    st.caption("Sorted by negative share, highest first.")
    st.altair_chart(sentiment_chart(table, segment_label), width="stretch")


def topics_tab(departments):
    table = topics()
    st.caption(
        "Top BERTopic themes per department (merged and hand-labelled; fit on the training split). "
        "Share is the percentage of the department's reviews assigned to the theme."
    )
    shown = sorted(table["department"].unique()) if not departments else [d for d in departments if d in set(table["department"])]
    kinds = st.multiselect("Theme kind", sorted(table["kind"].unique()))
    for dept in shown:
        rows = table[table["department"] == dept]
        if kinds:
            rows = rows[rows["kind"].isin(kinds)]
        if rows.empty:
            continue
        st.subheader(dept)
        chart = (
            alt.Chart(rows)
            .mark_bar(color="#0f766e")
            .encode(
                y=alt.Y("theme:N", sort="-x", title=None),
                x=alt.X("share_of_dept:Q", title="% of department reviews"),
                tooltip=["theme", "kind", "top_words", alt.Tooltip("reviews:Q", format=","),
                         alt.Tooltip("share_of_dept:Q", title="% of dept")],
            )
            .properties(height=max(80, 30 * len(rows)))
        )
        st.altair_chart(chart, width="stretch")


def classify_tab():
    st.caption("TF-IDF + logistic regression baseline (val macro-F1 0.62). DistilBERT weights are not in the repo.")
    text = st.text_area("Review text", placeholder="Love the colour, but it runs small so I sent it back.")
    if not st.button("Classify", type="primary") or not text.strip():
        return
    model = baseline_model()
    row = predict_reviews(model, [text]).iloc[0]
    if not row["usable"]:
        st.warning("No usable words after preprocessing, so no prediction.")
        return
    st.markdown(f"**Predicted: {row['predicted']}**")
    cols = st.columns(len(SENTIMENT_LABELS))
    for col, label in zip(cols, SENTIMENT_LABELS):
        col.metric(label, f"{row[f'p_{label}']:.2f}")
    terms = influential_terms(model, text, row["predicted"])
    if terms:
        st.write("Driven by: " + ", ".join(f"`{t}` ({w:+.2f})" for t, w in terms))


def main():
    st.title("E-commerce review analytics")
    df = reviews()
    filtered, departments = sidebar_filters(df)
    headline(filtered)
    sentiment, topic, classify = st.tabs(["Sentiment by segment", "Top topics", "Classify a review"])
    with sentiment:
        sentiment_tab(filtered)
    with topic:
        topics_tab(departments)
    with classify:
        classify_tab()


main()
