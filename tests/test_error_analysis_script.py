"""Tests for the helpers in `scripts/error_analysis.py`.

The analysis itself lives in `src.evaluation.error_analysis` and is tested in
`tests/evaluation/`. What is left here is input handling: the DistilBERT
predictions arrive as a hand-saved CSV from Colab, so a wrong length, a wrong
column or a stray label must fail loudly instead of silently misaligning rows.
"""

import pandas as pd
import pytest

from scripts.error_analysis import load_predictions, snippet


def write_preds(tmp_path, **columns):
    path = tmp_path / "preds.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def test_load_predictions_returns_labels_in_order(tmp_path):
    path = write_preds(tmp_path, predicted=["positive", "neutral", "negative"])
    assert load_predictions(path, 3) == ["positive", "neutral", "negative"]


def test_load_predictions_rejects_wrong_row_count(tmp_path):
    path = write_preds(tmp_path, predicted=["positive", "neutral"])
    with pytest.raises(ValueError, match="2 rows, validation has 3"):
        load_predictions(path, 3)


def test_load_predictions_rejects_missing_column(tmp_path):
    path = write_preds(tmp_path, label=["positive"])
    with pytest.raises(ValueError, match="'predicted' column"):
        load_predictions(path, 1)


def test_load_predictions_rejects_unknown_label(tmp_path):
    path = write_preds(tmp_path, predicted=["positive", "LABEL_1"])
    with pytest.raises(ValueError, match="LABEL_1"):
        load_predictions(path, 2)


def test_snippet_collapses_whitespace_and_escapes_pipes():
    assert snippet("a  b\n c | d") == "a b c / d"


def test_snippet_truncates_long_text():
    out = snippet("word " * 200, limit=20)
    assert len(out) == 20 and out.endswith("…")
