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


def paired_bootstrap(
    y_true,
    pred_a,
    pred_b,
    labels: list[str],
    n_boot: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap of the macro-F1 difference, model B minus model A.

    Resamples review indices with replacement and scores both models on the
    same resample, so the interval reflects how much the *difference* depends
    on which reviews happen to be in the test set. Returns the observed delta,
    the percentile interval, and `share_b_not_better`: the fraction of
    resamples where B does not beat A (a one-sided bootstrap p-value).
    """
    import numpy as np

    y_true, pred_a, pred_b = (np.asarray(v) for v in (y_true, pred_a, pred_b))
    if not len(y_true) == len(pred_a) == len(pred_b):
        raise ValueError("y_true, pred_a and pred_b must be the same length")

    def macro_f1(idx, pred):
        return f1_score(y_true[idx], pred[idx], labels=labels, average="macro", zero_division=0)

    everything = np.arange(len(y_true))
    observed = macro_f1(everything, pred_b) - macro_f1(everything, pred_a)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        deltas[i] = macro_f1(idx, pred_b) - macro_f1(idx, pred_a)
    tail = (1 - confidence) / 2
    return {
        "delta": float(observed),
        "ci_low": float(np.quantile(deltas, tail)),
        "ci_high": float(np.quantile(deltas, 1 - tail)),
        "share_b_not_better": float((deltas <= 0).mean()),
        "n_boot": n_boot,
    }
