import pandas as pd
import pytest

from src.dashboard.data import (
    AGE_BAND_COL,
    CLASS_COL,
    DEPARTMENT_COL,
    SPLIT_COL,
    UNKNOWN,
    age_band,
    filter_reviews,
    load_dashboard_data,
    load_topic_table,
    overview,
    sentiment_breakdown,
)
from src.models.baseline import LABEL_COL


def _split_csv(rows):
    base = {
        "Clothing ID": 1, "Title": "", "Review Text": "text", "Recommended IND": 1,
        "Positive Feedback Count": 0, "Division Name": "General", "Review Text Clean": "nice dress",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


@pytest.fixture
def processed_dir(tmp_path):
    _split_csv([
        {"Age": 30, "Rating": 5, "Department Name": "Tops", "Class Name": "Knits"},
        {"Age": 70, "Rating": 1, "Department Name": None, "Class Name": None},
        {"Age": 40, "Rating": 4, "Department Name": "Dresses", "Class Name": "Dresses",
         "Review Text Clean": "   "},
    ]).to_csv(tmp_path / "train.csv", index=False)
    _split_csv([
        {"Age": 24, "Rating": 3, "Department Name": "Tops", "Class Name": "Blouses"},
    ]).to_csv(tmp_path / "val.csv", index=False)
    return tmp_path


@pytest.fixture
def reviews():
    return pd.DataFrame({
        DEPARTMENT_COL: ["Tops"] * 4 + ["Dresses"] * 2 + ["Trend"],
        CLASS_COL: ["Knits", "Knits", "Blouses", "Blouses", "Dresses", "Dresses", "Trend"],
        AGE_BAND_COL: ["25-34", "35-44", "25-34", "65+", "35-44", "35-44", "18-24"],
        LABEL_COL: ["positive", "positive", "negative", "neutral", "negative", "positive", "positive"],
        "Rating": [5, 4, 1, 3, 2, 5, 5],
        "Recommended IND": [1, 1, 0, 1, 0, 1, 1],
    })


def test_age_band_edges():
    bands = age_band(pd.Series([18, 24, 25, 64, 65, 99, 17]))
    assert list(bands.astype(object)[:6]) == ["18-24", "18-24", "25-34", "55-64", "65+", "65+"]
    assert pd.isna(bands.iloc[6])
    assert bands.cat.ordered


def test_load_dashboard_data_combines_splits_and_drops_blank_text(processed_dir):
    df = load_dashboard_data(processed_dir=processed_dir)
    assert len(df) == 3  # blank-text train row dropped
    assert df[SPLIT_COL].tolist() == ["train", "train", "val"]
    assert df[LABEL_COL].tolist() == ["positive", "negative", "neutral"]
    assert df[AGE_BAND_COL].astype(str).tolist() == ["25-34", "65+", "18-24"]


def test_load_dashboard_data_fills_missing_segments(processed_dir):
    df = load_dashboard_data(processed_dir=processed_dir)
    assert df.loc[1, DEPARTMENT_COL] == UNKNOWN
    assert df.loc[1, CLASS_COL] == UNKNOWN


def test_load_dashboard_data_refuses_test_split(processed_dir):
    with pytest.raises(ValueError, match="held out"):
        load_dashboard_data(splits=("train", "test"), processed_dir=processed_dir)


def test_filter_reviews_combines_filters(reviews):
    out = filter_reviews(reviews, departments=["Tops"], age_bands=["25-34"])
    assert out[CLASS_COL].tolist() == ["Knits", "Blouses"]


def test_filter_reviews_empty_means_no_filter(reviews):
    assert len(filter_reviews(reviews, departments=[], classes=None)) == len(reviews)


def test_sentiment_breakdown_shares_and_order(reviews):
    out = sentiment_breakdown(reviews, DEPARTMENT_COL)
    # Dresses 50% negative, Tops 25%, Trend 0% -> highest negative rate first
    assert out["segment"].unique().tolist() == ["Dresses", "Tops", "Trend"]
    assert out.groupby("segment")["share"].sum().round(10).eq(1).all()
    tops = out[out["segment"] == "Tops"].set_index("sentiment")
    assert tops["reviews"].to_dict() == {"negative": 1, "neutral": 1, "positive": 2}
    assert tops["segment_total"].unique().tolist() == [4]


def test_sentiment_breakdown_fills_zero_counts(reviews):
    out = sentiment_breakdown(reviews, DEPARTMENT_COL)
    trend = out[out["segment"] == "Trend"].set_index("sentiment")["reviews"]
    assert trend.to_dict() == {"negative": 0, "neutral": 0, "positive": 1}


def test_sentiment_breakdown_min_reviews(reviews):
    out = sentiment_breakdown(reviews, DEPARTMENT_COL, min_reviews=2)
    assert set(out["segment"]) == {"Tops", "Dresses"}
    assert sentiment_breakdown(reviews, DEPARTMENT_COL, min_reviews=100).empty


def test_sentiment_breakdown_categorical_segment(reviews):
    reviews[AGE_BAND_COL] = age_band(pd.Series([30, 40, 30, 70, 40, 40, 20]))
    out = sentiment_breakdown(reviews, AGE_BAND_COL)
    assert set(out["segment"]) == {"18-24", "25-34", "35-44", "65+"}  # empty bands not shown


def test_overview(reviews):
    stats = overview(reviews)
    assert stats["reviews"] == 7
    assert stats["shares"] == pytest.approx({"negative": 2 / 7, "neutral": 1 / 7, "positive": 4 / 7})
    assert stats["recommend_rate"] == pytest.approx(5 / 7)
    assert stats["mean_rating"] == pytest.approx(25 / 7)


def test_overview_empty(reviews):
    stats = overview(reviews.iloc[0:0])
    assert stats["reviews"] == 0
    assert stats["shares"]["positive"] == 0.0


def test_load_topic_table_joins_kind(tmp_path):
    pd.DataFrame({
        "department": ["Tops", "Tops"], "topic": [1, 99], "reviews": [10, 5],
        "share_of_dept": [20.0, 10.0], "top_words": ["a, b", "c, d"], "theme": ["shirts", "new"],
    }).to_csv(tmp_path / "topics_merged_by_department.csv", index=False)
    pd.DataFrame({"topic": [1], "theme": ["shirts"], "kind": ["product"]}).to_csv(
        tmp_path / "topic_labels.csv", index=False)
    out = load_topic_table(tmp_path)
    assert out["kind"].tolist() == ["product", "unlabelled"]
    assert out["theme"].tolist() == ["shirts", "new"]


def test_load_topic_table_real_reports():
    out = load_topic_table()
    assert {"department", "theme", "kind", "share_of_dept"} <= set(out.columns)
    assert (out["kind"] != "unlabelled").all()
