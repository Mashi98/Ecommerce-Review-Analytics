"""Tests for the sweep driver's own logic.

`scripts/topic_sweep.py` is mostly a driver over `src.models.topics`, which is
covered by `tests/models/test_topics.py`. `md_table` is the exception: it is
new formatting logic, and its first run rendered missing values as "nan" and
integer counts as "224.0000", so it earns its own tests.
"""

import pandas as pd
import pytest

from scripts.topic_sweep import md_table


def cells(table: str) -> list[list[str]]:
    """Split a rendered markdown table into stripped cells, skipping the rule."""
    lines = table.splitlines()
    rows = [lines[0]] + lines[2:]
    return [[c.strip() for c in line.strip("|").split("|")] for line in rows]


def test_renders_header_and_values():
    table = md_table(pd.DataFrame({"model": ["LDA"], "n_topics": [5]}))
    assert cells(table) == [["model", "n_topics"], ["LDA", "5"]]


def test_separator_row_present():
    table = md_table(pd.DataFrame({"a": [1]}))
    assert set(table.splitlines()[1]) <= {"|", "-"}


def test_floats_use_the_default_format():
    table = md_table(pd.DataFrame({"coherence_cv": [0.446308]}))
    assert cells(table)[1] == ["0.4463"]


def test_nan_renders_blank_not_nan():
    """NaN is a float, so a naive isinstance check formats it as "nan"."""
    table = md_table(pd.DataFrame({"outlier_pct": [18.7, float("nan")]}))
    assert cells(table)[2] == [""]


def test_nullable_integer_keeps_integer_form():
    """Int64 carries the LDA gap without floating the BERTopic counts."""
    column = pd.array([224, None], dtype="Int64")
    table = md_table(pd.DataFrame({"outlier_docs": column}))
    assert [row[0] for row in cells(table)[1:]] == ["224", ""]


def test_none_renders_blank():
    table = md_table(pd.DataFrame({"note": ["seen", None]}))
    assert cells(table)[2] == [""]


def test_col_formats_override_per_column():
    frame = pd.DataFrame({"coherence_cv": [0.446308], "outlier_pct": [18.7]})
    table = md_table(frame, col_formats={"outlier_pct": "{:.1f}"})
    assert cells(table)[1] == ["0.4463", "18.7"]


def test_columns_are_width_aligned():
    table = md_table(pd.DataFrame({"model": ["LDA", "BERTopic"]}))
    widths = {len(line) for line in table.splitlines()}
    assert len(widths) == 1


def test_empty_frame_still_renders_its_header():
    table = md_table(pd.DataFrame({"model": [], "n_topics": []}))
    assert cells(table) == [["model", "n_topics"]]


@pytest.mark.parametrize("value", ["", "a | b"])
def test_string_values_pass_through(value):
    table = md_table(pd.DataFrame({"top_words": [value]}))
    assert value in table
