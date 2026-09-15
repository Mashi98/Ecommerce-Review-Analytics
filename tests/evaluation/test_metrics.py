import pandas as pd
import pytest

from src.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_df,
    evaluate,
    per_class_report,
)

LABELS = ["negative", "neutral", "positive"]


@pytest.fixture
def y_pair():
    y_true = ["negative", "negative", "neutral", "positive", "positive", "positive"]
    y_pred = ["negative", "neutral", "neutral", "positive", "positive", "negative"]
    return y_true, y_pred


def test_classification_metrics_perfect_prediction():
    y = ["negative", "neutral", "positive"]
    metrics = classification_metrics(y, y, LABELS)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_classification_metrics_computes_accuracy(y_pair):
    y_true, y_pred = y_pair
    metrics = classification_metrics(y_true, y_pred, LABELS)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_macro_f1_penalises_majority_only_prediction():
    """Predicting only the majority class should look good on accuracy
    but bad on macro-F1 — the whole reason macro-F1 is the headline metric."""
    y_true = ["positive"] * 8 + ["negative", "neutral"]
    y_pred = ["positive"] * 10
    metrics = classification_metrics(y_true, y_pred, LABELS)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["macro_f1"] < 0.4


def test_per_class_report_shape_and_support(y_pair):
    y_true, y_pred = y_pair
    report = per_class_report(y_true, y_pred, LABELS)
    assert list(report.index) == LABELS
    assert list(report.columns) == ["precision", "recall", "f1", "support"]
    assert report["support"].sum() == len(y_true)
    assert report.loc["negative", "support"] == 2


def test_confusion_matrix_df_orientation(y_pair):
    y_true, y_pred = y_pair
    matrix = confusion_matrix_df(y_true, y_pred, LABELS)
    assert matrix.shape == (3, 3)
    assert matrix.values.sum() == len(y_true)
    # one true "negative" was predicted "neutral"
    assert matrix.loc["negative", "neutral"] == 1
    assert matrix.loc["negative", "negative"] == 1


def test_evaluate_collects_all_sections():
    class StubModel:
        def predict(self, X):
            return ["positive"] * len(X)

    X = ["a", "b", "c"]
    y_true = ["positive", "positive", "negative"]
    result = evaluate(StubModel(), X, y_true, LABELS)
    assert set(result) == {"summary", "per_class", "confusion_matrix", "predictions"}
    assert result["summary"]["accuracy"] == pytest.approx(2 / 3)
    assert isinstance(result["per_class"], pd.DataFrame)
    assert len(result["predictions"]) == 3
