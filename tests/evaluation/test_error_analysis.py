import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    CONFIDENCE_COL,
    build_error_frame,
    disagreement,
    error_rate_by,
    error_rate_by_length,
    error_type_counts,
    high_confidence_errors,
    sample_errors,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "Sentiment": ["negative", "neutral", "neutral", "positive", "positive", "positive"],
            "Review Text": ["bad", "ok " * 30, "meh", "great", "love " * 60, "nice"],
            "Department Name": ["Tops", "Tops", "Dresses", "Dresses", "Dresses", "Tops"],
        }
    )


@pytest.fixture
def frame(df):
    pred = ["negative", "positive", "neutral", "positive", "neutral", "positive"]
    proba = [
        [0.9, 0.05, 0.05],
        [0.1, 0.1, 0.8],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
    ]
    return build_error_frame(df, pred, "Sentiment", "Review Text", proba=proba)


def test_build_error_frame_flags_correctness_and_error_type(frame):
    assert frame["correct"].tolist() == [True, False, True, True, False, True]
    assert frame.loc[1, "error_type"] == "neutral -> positive"
    assert frame.loc[4, "error_type"] == "positive -> neutral"


def test_build_error_frame_confidence_is_probability_of_predicted_class(frame):
    assert frame[CONFIDENCE_COL].tolist() == [0.9, 0.8, 0.6, 0.8, 0.7, 0.8]


def test_build_error_frame_counts_words(frame):
    assert frame["n_words"].tolist() == [1, 30, 1, 1, 60, 1]


def test_build_error_frame_does_not_mutate_input(df):
    before = df.copy()
    build_error_frame(df, ["negative"] * len(df), "Sentiment", "Review Text")
    pd.testing.assert_frame_equal(df, before)


def test_build_error_frame_rejects_length_mismatch(df):
    with pytest.raises(ValueError, match="predictions"):
        build_error_frame(df, ["negative"], "Sentiment", "Review Text")


def test_build_error_frame_without_proba_has_no_confidence(df):
    frame = build_error_frame(df, ["negative"] * len(df), "Sentiment", "Review Text")
    assert CONFIDENCE_COL not in frame


def test_error_type_counts_shares(frame):
    counts = error_type_counts(frame).set_index("error_type")
    assert counts["count"].to_dict() == {"neutral -> positive": 1, "positive -> neutral": 1}
    assert counts["share_of_errors"].sum() == pytest.approx(1.0)
    # 1 of the 2 neutral reviews went to positive; 1 of the 3 positive went to neutral.
    assert counts.loc["neutral -> positive", "share_of_class"] == pytest.approx(0.5)
    assert counts.loc["positive -> neutral", "share_of_class"] == pytest.approx(1 / 3)


def test_error_type_counts_empty_when_no_errors(df):
    frame = build_error_frame(df, df["Sentiment"].tolist(), "Sentiment", "Review Text")
    assert error_type_counts(frame).empty


def test_error_rate_by_group(frame):
    out = error_rate_by(frame, "Department Name").set_index("Department Name")
    assert out.loc["Tops", "n"] == 3 and out.loc["Tops", "errors"] == 1
    assert out.loc["Dresses", "error_rate"] == pytest.approx(1 / 3)


def test_error_rate_by_respects_min_support(frame):
    out = error_rate_by(frame, "Department Name", min_support=4)
    assert out.empty


def test_error_rate_by_length_buckets_in_order(frame):
    out = error_rate_by_length(frame)
    assert out["length"].tolist() == ["<=20", "21-50", "51-100"]
    by_bucket = out.set_index("length")
    assert by_bucket.loc["<=20", "n"] == 4
    assert by_bucket.loc["21-50", "error_rate"] == 1.0
    assert by_bucket.loc["51-100", "error_rate"] == 1.0


def test_high_confidence_errors_filters_and_sorts(frame):
    out = high_confidence_errors(frame, threshold=0.7)
    assert out[CONFIDENCE_COL].tolist() == [0.8, 0.7]
    assert not out["correct"].any()


def test_high_confidence_errors_needs_confidence(df):
    frame = build_error_frame(df, ["negative"] * len(df), "Sentiment", "Review Text")
    with pytest.raises(ValueError, match="confidence"):
        high_confidence_errors(frame)


def test_sample_errors_is_reproducible_and_capped(frame):
    a = sample_errors(frame, "neutral -> positive", n=5, seed=1)
    b = sample_errors(frame, "neutral -> positive", n=5, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 1  # only one such error exists


def test_disagreement_counts_each_outcome(df):
    a = build_error_frame(df, ["negative", "positive", "neutral", "positive", "neutral", "positive"], "Sentiment", "Review Text")
    b = build_error_frame(df, ["negative", "neutral", "positive", "positive", "neutral", "positive"], "Sentiment", "Review Text")
    out = disagreement(a, b).set_index("outcome")["count"].to_dict()
    # rows: 0 both right, 1 only B, 2 only A, 3 both right, 4 both wrong, 5 both right
    assert out == {"both right": 3, "only A right": 1, "only B right": 1, "both wrong": 1}


def test_disagreement_rejects_different_rows(df):
    a = build_error_frame(df, df["Sentiment"].tolist(), "Sentiment", "Review Text")
    with pytest.raises(ValueError, match="same rows"):
        disagreement(a, a.iloc[::-1].reset_index(drop=True))
