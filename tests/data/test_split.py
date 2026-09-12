import pandas as pd
import pytest

from src.data.split import add_preprocessed_text, clean_reviews, split_reviews


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Clothing ID": [1, 2, 3, 4, 5],
            "Age": [33, 40, 25, 50, 29],
            "Title": ["Great", None, "Ok", "Bad", "Love it"],
            "Review Text": ["Loved it", "It was okay", None, "Not great", "Amazing fit"],
            "Rating": [5, 3, 3, 2, 5],
            "Recommended IND": [1, 1, 1, 0, 1],
            "Positive Feedback Count": [0, 2, 1, 0, 3],
            "Division Name": ["General", "General", None, "General", "General"],
            "Department Name": ["Tops", "Tops", "Bottoms", "Tops", "Dresses"],
            "Class Name": ["Blouses", "Blouses", "Pants", "Blouses", "Dresses"],
        }
    )


def test_clean_reviews_drops_duplicates():
    df = pd.DataFrame(
        {
            "Clothing ID": [1, 1],
            "Division Name": ["General", "General"],
            "Department Name": ["Tops", "Tops"],
            "Class Name": ["Blouses", "Blouses"],
        }
    )
    cleaned = clean_reviews(df)
    assert len(cleaned) == 1


def test_clean_reviews_drops_missing_division_department_class(sample_df):
    cleaned = clean_reviews(sample_df)
    assert cleaned["Division Name"].isna().sum() == 0
    assert len(cleaned) == 4


def test_add_preprocessed_text_handles_missing_text(sample_df):
    cleaned = clean_reviews(sample_df)
    result = add_preprocessed_text(cleaned)
    assert "Review Text Clean" in result.columns
    assert (result["Review Text Clean"] != "").sum() >= 1


def test_split_reviews_fractions_sum_to_total():
    df = pd.DataFrame(
        {
            "Rating": [1, 2, 3, 4, 5] * 20,
            "value": range(100),
        }
    )
    train_df, val_df, test_df = split_reviews(df)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15


def test_split_reviews_no_overlap():
    df = pd.DataFrame(
        {
            "Rating": [1, 2, 3, 4, 5] * 20,
            "id": range(100),
        }
    )
    train_df, val_df, test_df = split_reviews(df)
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_reviews_invalid_fractions_raises():
    df = pd.DataFrame({"Rating": [1, 2, 3, 4, 5] * 20})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_reviews(df, train_frac=0.5, val_frac=0.3, test_frac=0.3)


def test_split_reviews_stratified_by_rating():
    df = pd.DataFrame(
        {
            "Rating": [1, 2, 3, 4, 5] * 20,
            "id": range(100),
        }
    )
    train_df, val_df, test_df = split_reviews(df)
    train_props = train_df["Rating"].value_counts(normalize=True).sort_index()
    full_props = df["Rating"].value_counts(normalize=True).sort_index()
    assert (train_props - full_props).abs().max() < 0.05
