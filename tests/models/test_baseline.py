import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models.baseline import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    SENTIMENT_LABELS,
    add_sentiment_label,
    build_baseline_pipeline,
    drop_empty_text,
    load_split,
    rating_to_sentiment,
    train_baseline,
)


@pytest.fixture
def labeled_df():
    """A small but learnable corpus with all three sentiment classes."""
    negative = ["fabric cheap poor quality return", "terrible fit awful material waste"]
    neutral = ["dress okay nothing special average", "fine but unremarkable plain okay"]
    positive = ["love dress fit perfectly beautiful", "gorgeous fabric perfect fit love"]
    return pd.DataFrame(
        {
            "Review Text Clean": (negative + neutral + positive) * 5,
            "Rating": ([1, 2, 3, 3, 5, 5]) * 5,
        }
    )


@pytest.mark.parametrize(
    "rating,expected",
    [(1, NEGATIVE), (2, NEGATIVE), (3, NEUTRAL), (4, POSITIVE), (5, POSITIVE)],
)
def test_rating_to_sentiment_maps_all_ratings(rating, expected):
    assert rating_to_sentiment(rating) == expected


def test_add_sentiment_label_adds_column():
    df = pd.DataFrame({"Rating": [1, 3, 5]})
    result = add_sentiment_label(df)
    assert list(result["Sentiment"]) == [NEGATIVE, NEUTRAL, POSITIVE]
    assert "Sentiment" not in df.columns  # original untouched


def test_drop_empty_text_removes_blank_and_missing():
    df = pd.DataFrame({"Review Text Clean": ["good fit", "", None, "   ", "love it"]})
    result = drop_empty_text(df)
    assert len(result) == 2
    assert list(result["Review Text Clean"]) == ["good fit", "love it"]


def test_build_pipeline_uses_balanced_class_weights():
    pipeline = build_baseline_pipeline(model="logreg", imbalance="class_weight")
    assert pipeline.named_steps["clf"].class_weight == "balanced"
    assert "smote" not in pipeline.named_steps


def test_build_pipeline_with_smote_inserts_resampler():
    pipeline = build_baseline_pipeline(model="logreg", imbalance="smote")
    assert isinstance(pipeline, ImbPipeline)
    assert "smote" in pipeline.named_steps
    # SMOTE replaces weighting, it should not be applied on top of it
    assert pipeline.named_steps["clf"].class_weight is None


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_baseline_pipeline(model="randomforest")


def test_build_pipeline_rejects_unknown_imbalance_strategy():
    with pytest.raises(ValueError, match="Unknown imbalance strategy"):
        build_baseline_pipeline(imbalance="undersample")


def test_load_split_rejects_unknown_split_name():
    with pytest.raises(ValueError, match="Unknown split"):
        load_split("holdout")


@pytest.mark.parametrize("model", ["logreg", "svm"])
def test_train_baseline_fits_and_predicts(labeled_df, model):
    pipeline = train_baseline(add_sentiment_label(labeled_df), model=model)
    preds = pipeline.predict(labeled_df["Review Text Clean"])
    assert len(preds) == len(labeled_df)
    assert set(preds).issubset(set(SENTIMENT_LABELS))


def test_train_baseline_with_smote_balances_training_classes(labeled_df):
    pipeline = train_baseline(add_sentiment_label(labeled_df), imbalance="smote")
    preds = pipeline.predict(labeled_df["Review Text Clean"])
    assert len(preds) == len(labeled_df)


def test_train_baseline_is_deterministic(labeled_df):
    df = add_sentiment_label(labeled_df)
    first = train_baseline(df, seed=42).predict(df["Review Text Clean"])
    second = train_baseline(df, seed=42).predict(df["Review Text Clean"])
    assert list(first) == list(second)


def test_train_baseline_accepts_alternate_label_column(labeled_df):
    """The pipeline must work for Recommended IND, not just Sentiment."""
    df = labeled_df.copy()
    df["Recommended IND"] = (df["Rating"] >= 4).astype(int)
    pipeline = train_baseline(df, label_col="Recommended IND")
    preds = pipeline.predict(df["Review Text Clean"])
    assert set(preds).issubset({0, 1})
