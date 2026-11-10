"""Data preparation for the Streamlit dashboard.

Everything the app shows is computed here so it can be unit-tested without
Streamlit. The app (`app/dashboard.py`) only wires these functions to widgets.

Scope: the dashboard reads the train and validation splits only. The test split
is held out for the single final evaluation, so it is not loaded here, not even
for descriptive counts. Sentiment is the rating-derived label (1-2 negative,
3 neutral, 4-5 positive), not a model prediction.

The dataset has no review dates, so "trends" means how sentiment shifts across
segments (department, class, age band), not over time.
"""

from pathlib import Path

import pandas as pd

from src.models.baseline import LABEL_COL, PROCESSED_DIR, SENTIMENT_LABELS, load_split

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

DASHBOARD_SPLITS = ("train", "val")

DEPARTMENT_COL = "Department Name"
CLASS_COL = "Class Name"
AGE_COL = "Age"
AGE_BAND_COL = "Age Band"
SPLIT_COL = "split"

AGE_BINS = [18, 25, 35, 45, 55, 65, 200]
AGE_BAND_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

SEGMENT_COLUMNS = {
    "Department": DEPARTMENT_COL,
    "Class": CLASS_COL,
    "Age band": AGE_BAND_COL,
}

UNKNOWN = "Unknown"


def age_band(ages: pd.Series) -> pd.Series:
    """Bucket ages into the fixed bands in AGE_BAND_LABELS (ordered categorical).

    Bands are left-closed: 25 falls in "25-34". Ages outside [18, 200) become NaN.
    """
    return pd.cut(ages, bins=AGE_BINS, labels=AGE_BAND_LABELS, right=False)


def load_dashboard_data(
    splits: tuple[str, ...] = DASHBOARD_SPLITS,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Labelled, text-filtered reviews from `splits`, with segment columns ready.

    Refuses the test split so the held-out evaluation set cannot leak into the
    dashboard by accident.
    """
    if "test" in splits:
        raise ValueError("The test split is held out and is not shown in the dashboard")
    frames = []
    for name in splits:
        part = load_split(name, processed_dir)
        part[SPLIT_COL] = name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    for col in (DEPARTMENT_COL, CLASS_COL):
        df[col] = df[col].fillna(UNKNOWN)
    df[AGE_BAND_COL] = age_band(df[AGE_COL])
    return df


def filter_reviews(
    df: pd.DataFrame,
    departments: list[str] | None = None,
    classes: list[str] | None = None,
    age_bands: list[str] | None = None,
) -> pd.DataFrame:
    """Keep rows matching every filter given. None or an empty list means no filter."""
    mask = pd.Series(True, index=df.index)
    for col, wanted in ((DEPARTMENT_COL, departments), (CLASS_COL, classes), (AGE_BAND_COL, age_bands)):
        if wanted:
            mask &= df[col].isin(wanted)
    return df[mask]


def sentiment_breakdown(df: pd.DataFrame, by: str, min_reviews: int = 1) -> pd.DataFrame:
    """Sentiment counts and within-segment shares, one row per (segment, sentiment).

    Columns: segment, sentiment, reviews, share (0-1, sums to 1 per segment),
    segment_total. Segments with fewer than `min_reviews` reviews are dropped;
    every sentiment appears for every kept segment, with zero counts filled in.
    Rows are ordered by segment negative rate, highest first, so the
    problem segments come first.
    """
    counts = (
        df.groupby([by, LABEL_COL], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_LABELS, fill_value=0)
    )
    totals = counts.sum(axis=1)
    counts = counts[totals >= min_reviews]
    totals = totals[totals >= min_reviews]
    if counts.empty:
        return pd.DataFrame(columns=["segment", "sentiment", "reviews", "share", "segment_total"])

    shares = counts.div(totals, axis=0)
    order = shares["negative"].sort_values(ascending=False, kind="stable").index

    long = counts.loc[order].stack().rename("reviews").reset_index()
    long.columns = ["segment", "sentiment", "reviews"]
    long["segment"] = long["segment"].astype(str)
    long["share"] = long["reviews"] / long["segment"].map(totals.rename(index=str))
    long["segment_total"] = long["segment"].map(totals.rename(index=str)).astype(int)
    return long


def overview(df: pd.DataFrame) -> dict:
    """Headline numbers: review count, sentiment shares, recommend rate, mean rating."""
    n = len(df)
    if n == 0:
        return {"reviews": 0, "shares": dict.fromkeys(SENTIMENT_LABELS, 0.0),
                "recommend_rate": float("nan"), "mean_rating": float("nan")}
    shares = df[LABEL_COL].value_counts(normalize=True).reindex(SENTIMENT_LABELS, fill_value=0.0)
    return {
        "reviews": n,
        "shares": {label: float(shares[label]) for label in SENTIMENT_LABELS},
        "recommend_rate": float(df["Recommended IND"].mean()),
        "mean_rating": float(df["Rating"].mean()),
    }


def load_topic_table(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    """Top merged BERTopic topics per department, with the hand-labelled topic kind.

    Joins `topics_merged_by_department.csv` (fit on the training split) with the
    `kind` column of `topic_labels.csv` (product, sizing-fit, issue, ...).
    Topics missing from the label file get kind "unlabelled".
    """
    topics = pd.read_csv(reports_dir / "topics_merged_by_department.csv")
    labels = pd.read_csv(reports_dir / "topic_labels.csv")[["topic", "kind"]]
    merged = topics.merge(labels, on="topic", how="left")
    merged["kind"] = merged["kind"].fillna("unlabelled")
    return merged
