"""Error analysis driver: where does a sentiment model go wrong on validation?

Runs on the validation split only — the test set is reserved for the single
final evaluation. With no arguments it fits the TF-IDF + LogReg baseline on
train and analyses its validation predictions. To analyse another model
(DistilBERT), save its validation predictions from Colab as a CSV with a
`predicted` column, one row per row of `load_split("val")` and in the same
order, and pass it in:

    python -m scripts.error_analysis
    python -m scripts.error_analysis --preds val_preds_distilbert.csv --name distilbert --against baseline

`--against` adds a row-by-row comparison with an earlier run, read from its
`reports/error_analysis_<name>_rows.csv`.

Writes:
  reports/error_analysis_<name>.md        the written breakdown
  reports/error_analysis_<name>_rows.csv  every validation row with its prediction
"""

import argparse
from pathlib import Path

import pandas as pd

from scripts.topic_sweep import md_table
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
from src.evaluation.metrics import classification_metrics, confusion_matrix_df, per_class_report
from src.models.baseline import LABEL_COL, SENTIMENT_LABELS, TEXT_COL, load_split, train_baseline
from src.models.transformer import TEXT_COL_RAW

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

MIN_GROUP_SUPPORT = 30
SAMPLES_PER_ERROR_TYPE = 8
HIGH_CONFIDENCE = 0.8
TOP_ERROR_TYPES_SAMPLED = 3
SNIPPET_CHARS = 280


def snippet(text, limit: int = SNIPPET_CHARS) -> str:
    """One-line excerpt safe to drop into a markdown table cell."""
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def sample_table(rows: pd.DataFrame) -> str:
    out = pd.DataFrame(
        {
            "rating": rows["Rating"],
            "true": rows["true"],
            "predicted": rows["predicted"],
            "review": rows[TEXT_COL_RAW].map(snippet),
        }
    )
    if CONFIDENCE_COL in rows:
        out.insert(3, "conf", rows[CONFIDENCE_COL].round(2))
    return md_table(out, col_formats={"conf": "{:.2f}"})


def baseline_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Baseline class predictions and probabilities on `val_df`."""
    model = train_baseline(train_df, model="logreg", imbalance="class_weight")
    proba = model.predict_proba(val_df[TEXT_COL])
    classes = list(model.classes_)
    return model.predict(val_df[TEXT_COL]).tolist(), pd.DataFrame(proba, columns=classes)


def load_predictions(path: Path, n_rows: int) -> list[str]:
    preds = pd.read_csv(path)
    if "predicted" not in preds:
        raise ValueError(f"{path} needs a 'predicted' column, has {list(preds.columns)}")
    if len(preds) != n_rows:
        raise ValueError(f"{path} has {len(preds)} rows, validation has {n_rows}")
    unknown = set(preds["predicted"]) - set(SENTIMENT_LABELS)
    if unknown:
        raise ValueError(f"{path} has labels outside {SENTIMENT_LABELS}: {sorted(unknown)}")
    return preds["predicted"].tolist()


def render_report(name: str, frame: pd.DataFrame, other: pd.DataFrame | None, other_name: str | None) -> str:
    y_true, y_pred = frame["true"].tolist(), frame["predicted"].tolist()
    summary = classification_metrics(y_true, y_pred, SENTIMENT_LABELS)
    lines = [f"# Error analysis — {name} (validation split)", ""]
    lines.append(
        f"{len(frame)} reviews, {int((~frame['correct']).sum())} misclassified. "
        + ", ".join(f"{k} {v:.4f}" for k, v in summary.items())
        + "."
    )

    lines += ["", "## Per-class", "", md_table(per_class_report(y_true, y_pred, SENTIMENT_LABELS).reset_index(names="class"))]
    lines += ["", "## Confusion matrix (rows true, columns predicted)", ""]
    lines.append(md_table(confusion_matrix_df(y_true, y_pred, SENTIMENT_LABELS).reset_index()))

    lines += ["", "## Error types", "", md_table(error_type_counts(frame), col_formats={"share_of_errors": "{:.3f}", "share_of_class": "{:.3f}"})]

    lines += ["", "## Error rate by star rating", ""]
    lines.append(md_table(error_rate_by(frame, "Rating").sort_values("Rating"), col_formats={"error_rate": "{:.3f}"}))
    lines += ["", "Labels come from ratings, so the 2-3 and 3-4 boundaries are where label noise is most likely.", ""]

    lines += ["## Error rate by review length (words)", "", md_table(error_rate_by_length(frame), col_formats={"error_rate": "{:.3f}"})]

    for col, title in [("Department Name", "department"), ("Class Name", "class")]:
        lines += ["", f"## Error rate by {title} (groups with at least {MIN_GROUP_SUPPORT} reviews)", ""]
        lines.append(md_table(error_rate_by(frame, col, MIN_GROUP_SUPPORT), col_formats={"error_rate": "{:.3f}"}))

    if CONFIDENCE_COL in frame:
        confident = high_confidence_errors(frame, HIGH_CONFIDENCE)
        lines += [
            "",
            f"## Confident errors (probability >= {HIGH_CONFIDENCE})",
            "",
            f"{len(confident)} errors ({len(confident) / max(int((~frame['correct']).sum()), 1):.0%} of all errors). "
            "The likeliest label-noise cases — read these first.",
            "",
            sample_table(confident.head(15)),
        ]

    top_types = error_type_counts(frame)["error_type"].head(TOP_ERROR_TYPES_SAMPLED)
    for error_type in top_types:
        lines += ["", f"## Sample: {error_type}", "", sample_table(sample_errors(frame, error_type, SAMPLES_PER_ERROR_TYPE))]

    if other is not None:
        lines += [
            "",
            f"## Disagreement with {other_name} (A = {name}, B = {other_name})",
            "",
            md_table(disagreement(frame, other)),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", type=Path, help="CSV with a 'predicted' column in validation row order")
    parser.add_argument("--name", default="baseline", help="label used in report and file names")
    parser.add_argument("--against", help="name of an earlier run to compare row-by-row")
    args = parser.parse_args()

    val_df = load_split("val")
    proba = None
    if args.preds:
        y_pred = load_predictions(args.preds, len(val_df))
    else:
        y_pred, proba = baseline_predictions(load_split("train"), val_df)

    frame = build_error_frame(val_df, y_pred, LABEL_COL, TEXT_COL_RAW, proba=proba)

    other = None
    if args.against:
        other_path = REPORTS_DIR / f"error_analysis_{args.against}_rows.csv"
        other_rows = pd.read_csv(other_path)
        other = build_error_frame(val_df, other_rows["predicted"].tolist(), LABEL_COL, TEXT_COL_RAW)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"error_analysis_{args.name}.md").write_text(
        render_report(args.name, frame, other, args.against), encoding="utf-8"
    )
    keep = ["Rating", "Department Name", "Class Name", TEXT_COL_RAW, "true", "predicted", "correct"]
    if CONFIDENCE_COL in frame:
        keep.append(CONFIDENCE_COL)
    frame[keep].to_csv(REPORTS_DIR / f"error_analysis_{args.name}_rows.csv", index=False)
    print(f"wrote reports/error_analysis_{args.name}.md")


if __name__ == "__main__":
    main()
