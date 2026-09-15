"""TF-IDF + linear classifier baseline for review sentiment.

Target label is 3-class sentiment derived from Rating (1-2 negative,
3 neutral, 4-5 positive). The training entry points take a `label_col`
so the same pipeline can be pointed at `Recommended IND` later without
changes.

Class imbalance (~78% positive) is handled either with balanced class
weights or with SMOTE oversampling on the TF-IDF features; SMOTE is
applied inside an imblearn Pipeline so it only ever runs at fit time and
never touches val/test.
"""

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features.vectorize import build_vectorizer

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

RANDOM_SEED = 42
TEXT_COL = "Review Text Clean"
LABEL_COL = "Sentiment"

NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"
SENTIMENT_LABELS = [NEGATIVE, NEUTRAL, POSITIVE]


def rating_to_sentiment(rating: int) -> str:
    """Map a 1-5 star rating onto a 3-class sentiment label."""
    if rating <= 2:
        return NEGATIVE
    if rating == 3:
        return NEUTRAL
    return POSITIVE


def add_sentiment_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `Sentiment` column derived from Rating."""
    df = df.copy()
    df[LABEL_COL] = df["Rating"].apply(rating_to_sentiment)
    return df


def drop_empty_text(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.DataFrame:
    """Drop rows whose preprocessed text is missing or blank.

    ~3.6% of the raw dataset has no `Review Text` (EDA finding); those rows
    carry no signal for a text classifier and are dropped for this task only.
    """
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    return df[df[text_col].str.strip() != ""].reset_index(drop=True)


def load_split(name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load one of the train/val/test CSVs, labeled and text-filtered."""
    if name not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {name!r} (expected train, val or test)")
    df = pd.read_csv(processed_dir / f"{name}.csv")
    return drop_empty_text(add_sentiment_label(df))


def build_baseline_pipeline(
    model: str = "logreg",
    imbalance: str = "class_weight",
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """TF-IDF -> linear classifier pipeline.

    `model` is "logreg" (LogisticRegression) or "svm" (LinearSVC).
    `imbalance` is "class_weight" (balanced weights), "smote" (oversample the
    TF-IDF features) or "none".
    """
    if imbalance not in {"class_weight", "smote", "none"}:
        raise ValueError(f"Unknown imbalance strategy: {imbalance!r}")

    class_weight = "balanced" if imbalance == "class_weight" else None

    if model == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=seed,
        )
    elif model == "svm":
        clf = LinearSVC(class_weight=class_weight, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model!r} (expected 'logreg' or 'svm')")

    steps = [("tfidf", build_vectorizer())]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", clf))

    # imblearn's Pipeline is needed for SMOTE (resampling steps are skipped at
    # predict time); it is a drop-in superset of sklearn's for the other cases.
    return ImbPipeline(steps) if imbalance == "smote" else Pipeline(steps)


def train_baseline(
    train_df: pd.DataFrame,
    model: str = "logreg",
    imbalance: str = "class_weight",
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Fit a baseline pipeline on a training split."""
    pipeline = build_baseline_pipeline(model=model, imbalance=imbalance, seed=seed)
    pipeline.fit(train_df[text_col], train_df[label_col])
    return pipeline


MODELS = ["logreg", "svm"]
IMBALANCE_STRATEGIES = ["class_weight", "smote", "none"]


def compare_baselines(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    models: list[str] = None,
    strategies: list[str] = None,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    labels: list[str] = None,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Train every model x imbalance-strategy combination and score them.

    Returns a summary DataFrame sorted by macro-F1 (the headline metric —
    accuracy flatters the majority class) and a dict of the fitted pipelines
    with their full evaluation output, keyed by (model, strategy).
    """
    from src.evaluation.metrics import evaluate

    models = models or MODELS
    strategies = strategies or IMBALANCE_STRATEGIES
    labels = labels or SENTIMENT_LABELS

    rows, results = [], {}
    for model in models:
        for strategy in strategies:
            pipeline = train_baseline(
                train_df,
                model=model,
                imbalance=strategy,
                text_col=text_col,
                label_col=label_col,
                seed=seed,
            )
            result = evaluate(pipeline, eval_df[text_col], eval_df[label_col], labels)
            results[(model, strategy)] = {"pipeline": pipeline, **result}
            rows.append({"model": model, "imbalance": strategy, **result["summary"]})

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return summary, results


if __name__ == "__main__":
    train_split, val_split = load_split("train"), load_split("val")
    print(f"train: {len(train_split)} rows | val: {len(val_split)} rows")
    print(f"train class balance: {train_split[LABEL_COL].value_counts().to_dict()}\n")

    summary_df, all_results = compare_baselines(train_split, val_split)
    print("Validation results (sorted by macro-F1):")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_key = (summary_df.loc[0, "model"], summary_df.loc[0, "imbalance"])
    best = all_results[best_key]
    print(f"\nBest: {best_key[0]} / {best_key[1]}")
    print(f"\nPer-class metrics:\n{best['per_class'].to_string()}")
    print(f"\nConfusion matrix:\n{best['confusion_matrix'].to_string()}")
