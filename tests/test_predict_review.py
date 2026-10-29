"""Tests for `scripts/predict_review.py`.

The model is a real baseline pipeline fitted on a tiny synthetic corpus, so the
tests exercise the same TF-IDF + logistic regression code path as the script
without loading the project's data splits.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.predict_review import format_prediction, influential_terms, predict_reviews, read_reviews
from src.models.baseline import LABEL_COL, TEXT_COL, train_baseline


@pytest.fixture(scope="module")
def model():
    texts = (
        ["love lovely great perfect beautiful"] * 12
        + ["return awful terrible poor cheap"] * 12
        + ["fine okay average decent"] * 12
    )
    labels = ["positive"] * 12 + ["negative"] * 12 + ["neutral"] * 12
    train = pd.DataFrame({TEXT_COL: texts, LABEL_COL: labels})
    return train_baseline(train, model="logreg", imbalance="class_weight")


def test_predict_reviews_returns_one_row_per_review(model):
    out = predict_reviews(model, ["I love this, it is lovely", "Awful and cheap, I will return it"])
    assert out["review"].tolist() == ["I love this, it is lovely", "Awful and cheap, I will return it"]
    assert out["predicted"].tolist() == ["positive", "negative"]


def test_predict_reviews_probabilities_sum_to_one_and_match_prediction(model):
    out = predict_reviews(model, ["great and perfect", "okay, average"])
    proba = out[["p_negative", "p_neutral", "p_positive"]]
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert out["predicted"].tolist() == [c.removeprefix("p_") for c in proba.idxmax(axis=1)]


def test_predict_reviews_flags_text_with_no_usable_words(model):
    out = predict_reviews(model, ["   ", "12345 !!!", "great"])
    assert out["usable"].tolist() == [False, False, True]


def test_influential_terms_are_review_words_that_support_the_label(model):
    terms = influential_terms(model, "Lovely and perfect, but the box was blue", "positive")
    names = [name for name, _ in terms]
    assert "lovely" in names and "perfect" in names
    assert "blue" not in names  # unknown to the vectorizer, cannot contribute
    assert all(weight > 0 for _, weight in terms)
    assert [w for _, w in terms] == sorted((w for _, w in terms), reverse=True)


def test_influential_terms_respects_limit(model):
    assert len(influential_terms(model, "love lovely great perfect beautiful", "positive", n=2)) == 2


def test_influential_terms_empty_when_nothing_matches(model):
    assert influential_terms(model, "zzz qqq", "positive") == []


def test_format_prediction_shows_label_probabilities_and_terms(model):
    row = predict_reviews(model, ["love it"]).iloc[0]
    text = format_prediction(row, [("love", 0.42)])
    assert text.startswith("positive")
    assert "negative " in text and "neutral " in text and "positive " in text
    assert "love (+0.42)" in text


def test_format_prediction_refuses_to_predict_without_words(model):
    row = predict_reviews(model, ["???"]).iloc[0]
    assert "no prediction" in format_prediction(row, [])


def test_read_reviews_skips_blank_lines_and_strips(tmp_path):
    path = tmp_path / "reviews.txt"
    path.write_text("  first review  \n\n\nsecond review\n   \n", encoding="utf-8")
    assert read_reviews(path) == ["first review", "second review"]
