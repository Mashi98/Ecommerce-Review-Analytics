import pandas as pd
import pytest

from src.data.load import EXPECTED_COLUMNS, load_reviews, schema_report


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Clothing ID": [1, 2, 2],
            "Age": [33, 40, 40],
            "Title": ["Great", None, None],
            "Review Text": ["Loved it", "It was okay", "It was okay"],
            "Rating": [5, 3, 3],
            "Recommended IND": [1, 1, 1],
            "Positive Feedback Count": [0, 2, 2],
            "Division Name": ["General", "General", "General"],
            "Department Name": ["Tops", "Tops", "Tops"],
            "Class Name": ["Blouses", "Blouses", "Blouses"],
        }
    )


def test_load_reviews_has_expected_columns():
    df = load_reviews()
    assert set(EXPECTED_COLUMNS).issubset(df.columns)


def test_load_reviews_nonempty():
    df = load_reviews()
    assert len(df) > 0


def test_load_reviews_missing_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"Clothing ID": [1], "Rating": [5]}).to_csv(bad_csv)
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_reviews(bad_csv)


def test_schema_report_counts_nulls(sample_df):
    report = schema_report(sample_df)
    assert report.loc["Title", "n_nulls"] == 2
    assert report.loc["Clothing ID", "n_nulls"] == 0


def test_schema_report_counts_duplicates(sample_df):
    report = schema_report(sample_df)
    assert report.attrs["n_duplicate_rows"] == 1
    assert report.attrs["n_rows"] == 3
