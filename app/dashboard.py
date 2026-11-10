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
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scripts.predict_review import influential_terms, predict_reviews  # noqa: E402
from src.dashboard.data import (  # noqa: E402
    AGE_BAND_LABELS,
    CLASS_COL,
    DEPARTMENT_COL,
    SEGMENT_COLUMNS,
    TOPIC_COL,
    attach_topics,
    filter_reviews,
    load_dashboard_data,
    load_topic_assignments,
    load_topic_labels,
    load_topic_table,
    overview,
    sentiment_breakdown,
    topic_sentiment_table,
)
from src.models.baseline import SENTIMENT_LABELS, load_split, train_baseline  # noqa: E402

SENTIMENT_COLORS = {"negative": "#c2410c", "neutral": "#a1a1aa", "positive": "#0f766e"}
SENTIMENT_SCALE = alt.Scale(domain=SENTIMENT_LABELS, range=[SENTIMENT_COLORS[s] for s in SENTIMENT_LABELS])
LABEL_LIMIT = 400  # px; Vega-Lite truncates axis labels at 180 by default, longest theme is ~300

st.set_page_config(page_title="Review Analytics", layout="wide")


@st.cache_data
def reviews():
    """Dashboard reviews, with per-review topics when they have been exported."""
    df = load_dashboard_data()
    assignments = load_topic_assignments()
    if assignments is None:
        return df, False
    return attach_topics(df, assignments, load_topic_labels()), True


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
            y=alt.Y("segment:N", sort=None, title=segment_label, axis=alt.Axis(labelLimit=LABEL_LIMIT)),
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


def theme_sentiment_section(df):
    st.subheader("Themes by sentiment")
    tagged = df[df[TOPIC_COL].notna() & (df[TOPIC_COL] != -1)]
    st.caption(
        f"Training-split reviews under the current filters that BERTopic placed in a theme "
        f"({len(tagged):,} reviews; validation reviews and unclustered reviews are left out). "
        "Each bar is one theme's sentiment mix."
    )
    left, mid, right = st.columns([1, 1, 1])
    rank_by = left.radio("Rank themes by", ["Negative share", "Review count"], horizontal=True)
    kinds = mid.multiselect("Theme kind", sorted(tagged["kind"].dropna().unique()), key="theme_kind")
    min_reviews = right.slider("Hide themes with fewer reviews than", 1, 200, 25, key="theme_min")  # 25 = BERTopic min_topic_size
    if kinds:
        tagged = tagged[tagged["kind"].isin(kinds)]
    table = topic_sentiment_table(tagged, min_reviews=min_reviews)
    if table.empty:
        st.info("No theme has enough reviews under the current filters.")
        return
    if rank_by == "Review count":
        table = table.sort_values("reviews", ascending=False, kind="stable")

    baseline = overview(tagged)["shares"]["negative"]
    st.caption(f"Negative share across all themed reviews shown: {baseline:.1%}. Themes above that are the problem areas.")

    long = table.melt(
        id_vars=["theme", "kind", "reviews", "share_of_reviews"],
        value_vars=SENTIMENT_LABELS, var_name="sentiment", value_name="share",
    )
    order = table["theme"].tolist()
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("theme:N", sort=order, title=None, axis=alt.Axis(labelLimit=LABEL_LIMIT)),
            x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of the theme's reviews"),
            color=alt.Color("sentiment:N", scale=SENTIMENT_SCALE, sort=SENTIMENT_LABELS, title=None),
            order=alt.Order("sentiment_order:Q"),
            tooltip=[
                "theme:N", "kind:N", "sentiment:N",
                alt.Tooltip("share:Q", format=".1%"),
                alt.Tooltip("reviews:Q", title="Theme reviews", format=","),
                alt.Tooltip("share_of_reviews:Q", title="Share of themed reviews", format=".1%"),
            ],
        )
        .transform_calculate(sentiment_order=f"indexof({SENTIMENT_LABELS!r}, datum.sentiment)")
        .properties(height=max(120, 24 * len(order)))
    )
    rule = alt.Chart(pd.DataFrame({"x": [baseline]})).mark_rule(strokeDash=[4, 3], color="#18181b").encode(x="x:Q")
    st.altair_chart(chart + rule, width="stretch")


def topics_tab(df, departments, has_topics):
    if has_topics:
        theme_sentiment_section(df)
    else:
        st.info("Per-review topics not exported yet: run `python -m scripts.topic_merge` to enable the theme-by-sentiment view.")
    st.subheader("Top themes per department")
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
                y=alt.Y("theme:N", sort="-x", title=None, axis=alt.Axis(labelLimit=LABEL_LIMIT)),
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
    df, has_topics = reviews()
    filtered, departments = sidebar_filters(df)
    headline(filtered)
    sentiment, topic, classify = st.tabs(["Sentiment by segment", "Topics", "Classify a review"])
    with sentiment:
        sentiment_tab(filtered)
    with topic:
        topics_tab(filtered, departments, has_topics)
    with classify:
        classify_tab()


main()
