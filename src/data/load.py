"""Load and schema-check the raw Women's E-Commerce Clothing Reviews dataset."""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "reviews.csv"

EXPECTED_COLUMNS = [
    "Clothing ID",
    "Age",
    "Title",
    "Review Text",
    "Rating",
    "Recommended IND",
    "Positive Feedback Count",
    "Division Name",
    "Department Name",
    "Class Name",
]


def load_reviews(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw reviews CSV into a DataFrame.

    Drops the unnamed index column the source CSV ships with and validates
    that all expected columns are present.
    """
    df = pd.read_csv(path, index_col=0)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df


def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, null count/rate, and duplicate-row count summary."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_nulls": df.isna().sum(),
            "null_rate": (df.isna().mean()).round(4),
        }
    )
    report.attrs["n_duplicate_rows"] = int(df.duplicated().sum())
    report.attrs["n_rows"] = len(df)
    return report
