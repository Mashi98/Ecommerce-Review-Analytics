"""Clean and split the reviews dataset into stratified train/val/test sets.

Cleaning applied before splitting (per EDA findings in Brain/progress):
  - drop exact duplicate rows
  - drop rows missing Division/Department/Class Name

Split is 70/15/15 (train/val/test), stratified by Rating, with a fixed
random seed for reproducibility.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_reviews
from src.features.preprocessing import preprocess

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and rows missing Division/Department/Class Name."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["Division Name", "Department Name", "Class Name"])
    return df.reset_index(drop=True)


def add_preprocessed_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `Review Text Clean` column with the tokenized/lemmatized text.

    Rows with missing/blank `Review Text` get an empty string.
    """
    df = df.copy()
    df["Review Text Clean"] = df["Review Text"].apply(preprocess)
    return df


def split_reviews(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split by Rating."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df["Rating"],
        random_state=seed,
    )
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_frac,
        stratify=temp_df["Rating"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_and_save_splits(output_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Load, clean, preprocess, split, and write train/val/test CSVs."""
    df = load_reviews()
    df = clean_reviews(df)
    df = add_preprocessed_text(df)
    train_df, val_df, test_df = split_reviews(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        split_df.to_csv(output_dir / f"{name}.csv", index=False)
    return splits


if __name__ == "__main__":
    result = build_and_save_splits()
    for name, split_df in result.items():
        print(f"{name}: {len(split_df)} rows -> {PROCESSED_DIR / f'{name}.csv'}")
