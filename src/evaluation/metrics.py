"""Classification metrics for the sentiment models.

Reports macro-averaged and per-class metrics alongside accuracy — with a
~78% positive majority class, accuracy alone is misleading, so macro-F1 is
the headline number used for model comparison.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict[str, float]:
    """Accuracy plus macro- and weighted-averaged F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def per_class_report(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support, indexed by label."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "precision": precision.round(4),
            "recall": recall.round(4),
            "f1": f1.round(4),
            "support": support,
        },
        index=labels,
    )


def confusion_matrix_df(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame (rows = true, cols = predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )


def evaluate(model, X, y_true, labels: list[str]) -> dict:
    """Run a fitted model over a split and collect every metric at once."""
    y_pred = model.predict(X)
    return {
        "summary": classification_metrics(y_true, y_pred, labels),
        "per_class": per_class_report(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix_df(y_true, y_pred, labels),
        "predictions": y_pred,
    }
