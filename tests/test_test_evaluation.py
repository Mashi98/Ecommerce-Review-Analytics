import pandas as pd
import pytest

from scripts.test_evaluation import (
    build_report,
    check_alignment,
    check_baseline_reproduces,
    per_class_table,
    summary_table,
)
from src.evaluation.metrics import classification_metrics
from src.models.baseline import SENTIMENT_LABELS
from src.models.topics import text_digest


@pytest.fixture
def split():
    return pd.DataFrame({"Review Text": ["love it", "runs small", "awful"]})


@pytest.fixture
def preds(split):
    return pd.DataFrame({
        "text_digest": [text_digest(t) for t in split["Review Text"]],
        "predicted": ["positive", "neutral", "negative"],
    })


def test_check_alignment_returns_predictions(preds, split):
    assert check_alignment(preds, split, "test") == ["positive", "neutral", "negative"]


def test_check_alignment_rejects_row_count(preds, split):
    with pytest.raises(ValueError, match="test: 2 predictions for 3 reviews"):
        check_alignment(preds.iloc[:2], split, "test")


def test_check_alignment_rejects_reordered_rows(preds, split):
    with pytest.raises(ValueError, match="2 rows do not match"):
        check_alignment(preds.iloc[[1, 0, 2]].reset_index(drop=True), split, "test")


def test_check_alignment_rejects_unknown_labels(preds, split):
    preds.loc[0, "predicted"] = "LABEL_2"
    with pytest.raises(ValueError, match="unknown labels"):
        check_alignment(preds, split, "test")


def test_check_baseline_reproduces():
    check_baseline_reproduces(0.62449, expected=0.6245)
    with pytest.raises(RuntimeError, match="0.6201"):
        check_baseline_reproduces(0.6201, expected=0.6245)


def test_summary_table():
    scores = {("baseline", "test"): {"accuracy": 0.8, "macro_f1": 0.6, "weighted_f1": 0.7}}
    out = summary_table(scores)
    assert out.columns.tolist() == ["model", "split", "accuracy", "macro_f1", "weighted_f1"]
    assert out.iloc[0].tolist() == ["baseline", "test", 0.8, 0.6, 0.7]


def test_per_class_table_side_by_side():
    y = ["negative", "neutral", "positive", "positive"]
    out = per_class_table(y, {"a": y, "b": ["positive"] * 4})
    assert out.columns.tolist() == ["class", "a P / R / F1", "b P / R / F1", "support"]
    assert out.loc[0, "a P / R / F1"] == "1.000 / 1.000 / 1.000"
    assert out["support"].tolist() == [1, 1, 2]


def test_build_report_mentions_key_numbers():
    y = ["negative", "neutral", "positive", "positive"]
    test_preds = {"baseline": ["positive"] * 4, "distilbert": y}
    scores = {("baseline", "test"): classification_metrics(y, test_preds["baseline"], SENTIMENT_LABELS)}
    boot = {"delta": 0.7333, "ci_low": 0.5, "ci_high": 0.9, "share_b_not_better": 0.0, "n_boot": 2000}
    text = build_report(scores, y, test_preds, boot)
    assert "**+0.7333**" in text
    assert "[+0.5000, +0.9000]" in text
    assert "Confusion matrix (test): distilbert" in text
    assert "held-out test split (4 reviews)" in text
