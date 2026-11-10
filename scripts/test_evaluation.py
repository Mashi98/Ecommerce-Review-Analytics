"""Final test-set evaluation: baseline vs DistilBERT, scored once.

DistilBERT is trained on Colab (`notebooks/02_distilbert_colab.ipynb`, section 6),
which exports its predictions. This script does the rest locally, in the pinned
environment, so the baseline number is reproducible on any machine:

  1. checks the exported predictions line up with the local val and test splits
     (row count and a digest of each review's text),
  2. trains the baseline and checks it still reproduces its validation result,
  3. scores both models on validation and test, with a paired bootstrap interval
     for the test macro-F1 difference.

    python -m scripts.test_evaluation

Reads:
  reports/val_preds_distilbert_final.csv   from Colab: text_digest, predicted
  reports/test_preds_distilbert.csv        from Colab: text_digest, predicted
Writes:
  reports/test_results.md
  reports/test_preds_baseline.csv          text_digest, predicted
"""

from pathlib import Path

import pandas as pd

from scripts.topic_sweep import md_table
from src.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_df,
    paired_bootstrap,
    per_class_report,
)
from src.models.baseline import LABEL_COL, SENTIMENT_LABELS, TEXT_COL, load_split, train_baseline
from src.models.topics import text_digest
from src.models.transformer import TEXT_COL_RAW

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
VAL_PREDS = "val_preds_distilbert_final.csv"
TEST_PREDS = "test_preds_distilbert.csv"

# The baseline's reported validation macro-F1. If retraining no longer gives
# this, the environment or data has drifted and the comparison is not valid.
BASELINE_VAL_MACRO_F1 = 0.6245


def check_alignment(preds: pd.DataFrame, split: pd.DataFrame, name: str) -> list[str]:
    """Return `preds["predicted"]` as a list, after checking it matches `split` row for row."""
    if len(preds) != len(split):
        raise ValueError(f"{name}: {len(preds)} predictions for {len(split)} reviews")
    expected = split[TEXT_COL_RAW].map(text_digest).to_numpy()
    mismatched = int((preds["text_digest"].to_numpy() != expected).sum())
    if mismatched:
        raise ValueError(f"{name}: {mismatched} rows do not match the local split's review text")
    unknown = set(preds["predicted"]) - set(SENTIMENT_LABELS)
    if unknown:
        raise ValueError(f"{name}: unknown labels {sorted(unknown)}")
    return preds["predicted"].tolist()


def check_baseline_reproduces(val_macro_f1: float, expected: float = BASELINE_VAL_MACRO_F1) -> None:
    if round(val_macro_f1, 4) != expected:
        raise RuntimeError(
            f"baseline validation macro-F1 is {val_macro_f1:.4f}, expected {expected}: "
            "the environment or the splits have changed since the reported result"
        )


def summary_table(scores: dict[tuple[str, str], dict[str, float]]) -> pd.DataFrame:
    """One row per (model, split) with accuracy, macro-F1 and weighted-F1."""
    rows = [{"model": model, "split": split, **metrics} for (model, split), metrics in scores.items()]
    return pd.DataFrame(rows)[["model", "split", "accuracy", "macro_f1", "weighted_f1"]]


def per_class_table(y_true, preds: dict[str, list[str]]) -> pd.DataFrame:
    """Per-class P / R / F1 for each model side by side, with support."""
    out = None
    for model, pred in preds.items():
        report = per_class_report(y_true, pred, SENTIMENT_LABELS)
        cell = report.apply(lambda r: f"{r.precision:.3f} / {r.recall:.3f} / {r.f1:.3f}", axis=1)
        frame = cell.rename(f"{model} P / R / F1").to_frame()
        out = frame if out is None else out.join(frame)
        support = report["support"]
    out["support"] = support
    return out.rename_axis("class").reset_index()


def build_report(
    scores: dict,
    y_test,
    test_preds: dict[str, list[str]],
    bootstrap: dict[str, float],
) -> str:
    base, bert = (classification_metrics(y_test, test_preds[m], SENTIMENT_LABELS) for m in ("baseline", "distilbert"))
    lines = [
        "# Final test-set evaluation",
        "",
        f"Both models scored once on the held-out test split ({len(y_test):,} reviews). "
        "DistilBERT predictions come from the final Colab run (weights saved to Drive); "
        "the baseline is retrained locally with the reported settings and checked to reproduce "
        f"its validation macro-F1 of {BASELINE_VAL_MACRO_F1} first.",
        "",
        "## Summary",
        "",
        md_table(summary_table(scores)),
        "",
        f"Test macro-F1: baseline {base['macro_f1']:.4f}, DistilBERT {bert['macro_f1']:.4f}.",
        "",
        "## Is the gain real?",
        "",
        f"Paired bootstrap over test reviews ({bootstrap['n_boot']:,} resamples): DistilBERT minus baseline "
        f"macro-F1 = **{bootstrap['delta']:+.4f}**, 95% interval "
        f"[{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}]. "
        f"DistilBERT failed to beat the baseline in {bootstrap['share_b_not_better']:.1%} of resamples.",
        "",
        "## Per class (test)",
        "",
        md_table(per_class_table(y_test, test_preds)),
        "",
    ]
    for model, pred in test_preds.items():
        lines += [f"## Confusion matrix (test): {model}", "", md_table(confusion_matrix_df(y_test, pred, SENTIMENT_LABELS).reset_index()), ""]
    return "\n".join(lines)


def main() -> None:
    for name in (VAL_PREDS, TEST_PREDS):
        if not (REPORTS_DIR / name).exists():
            raise SystemExit(f"missing reports/{name}: download it from the Colab run first (see docs/colab-setup.md)")

    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")
    bert_val = check_alignment(pd.read_csv(REPORTS_DIR / VAL_PREDS), val_df, "validation")
    bert_test = check_alignment(pd.read_csv(REPORTS_DIR / TEST_PREDS), test_df, "test")

    baseline = train_baseline(train_df, model="logreg", imbalance="class_weight")
    base_val = baseline.predict(val_df[TEXT_COL]).tolist()
    y_val, y_test = val_df[LABEL_COL].tolist(), test_df[LABEL_COL].tolist()
    check_baseline_reproduces(classification_metrics(y_val, base_val, SENTIMENT_LABELS)["macro_f1"])
    base_test = baseline.predict(test_df[TEXT_COL]).tolist()

    scores = {
        ("baseline", "validation"): classification_metrics(y_val, base_val, SENTIMENT_LABELS),
        ("baseline", "test"): classification_metrics(y_test, base_test, SENTIMENT_LABELS),
        ("distilbert", "validation"): classification_metrics(y_val, bert_val, SENTIMENT_LABELS),
        ("distilbert", "test"): classification_metrics(y_test, bert_test, SENTIMENT_LABELS),
    }
    test_preds = {"baseline": base_test, "distilbert": bert_test}
    bootstrap = paired_bootstrap(y_test, base_test, bert_test, SENTIMENT_LABELS)

    (REPORTS_DIR / "test_results.md").write_text(build_report(scores, y_test, test_preds, bootstrap), encoding="utf-8")
    pd.DataFrame({"text_digest": test_df[TEXT_COL_RAW].map(text_digest), "predicted": base_test}).to_csv(
        REPORTS_DIR / "test_preds_baseline.csv", index=False
    )
    print(summary_table(scores).round(4).to_string(index=False))
    print(f"\nbootstrap delta {bootstrap['delta']:+.4f} [{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}]")
    print("wrote reports/test_results.md, reports/test_preds_baseline.csv")


if __name__ == "__main__":
    main()
