"""Error analysis for the sentiment models.

Aggregate metrics say how often a model is wrong; this module says where. It
works on one frame per model — the evaluation split plus the model's
predictions — so the TF-IDF baseline and DistilBERT can be sliced the same
way and compared side by side.

Sentiment labels are derived from star ratings (1-2 negative, 3 neutral,
4-5 positive), so a "wrong" prediction is sometimes a noisy label rather than
a model failure: a 3-star review that reads warmly is labeled neutral. The
per-rating error rate and the confidence column are there to tell the two
apart — errors the model makes with high confidence are the first place to
look for label noise.
"""

import pandas as pd

TRUE_COL = "true"
PRED_COL = "predicted"
CORRECT_COL = "correct"
ERROR_TYPE_COL = "error_type"
CONFIDENCE_COL = "confidence"

# Review length buckets, in whitespace-separated words of the raw text.
LENGTH_BINS = [0, 20, 50, 100, float("inf")]
LENGTH_LABELS = ["<=20", "21-50", "51-100", ">100"]


def build_error_frame(
    df: pd.DataFrame,
    y_pred,
    label_col: str,
    text_col: str,
    proba=None,
) -> pd.DataFrame:
    """Attach predictions to the split they were made on.

    `y_pred` must be in `df` row order — both models predict over the frame
    returned by `load_split`, so that holds as long as the split is not
    re-filtered in between. `proba` (rows x classes), if given, is reduced to
    the probability of the predicted class.
    """
    y_pred = list(y_pred)
    if len(y_pred) != len(df):
        raise ValueError(f"{len(y_pred)} predictions for {len(df)} rows")

    frame = df.copy().reset_index(drop=True)
    frame[TRUE_COL] = frame[label_col]
    frame[PRED_COL] = y_pred
    frame[CORRECT_COL] = frame[TRUE_COL] == frame[PRED_COL]
    frame[ERROR_TYPE_COL] = frame[TRUE_COL] + " -> " + frame[PRED_COL]
    frame["n_words"] = frame[text_col].fillna("").astype(str).str.split().str.len()
    if proba is not None:
        proba = pd.DataFrame(proba).reset_index(drop=True)
        if len(proba) != len(frame):
            raise ValueError(f"{len(proba)} probability rows for {len(frame)} rows")
        frame[CONFIDENCE_COL] = proba.max(axis=1).to_numpy()
    return frame


def error_type_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """How many errors of each true -> predicted kind, most frequent first.

    `share_of_errors` sums to 1 over the wrong predictions; `share_of_class` is
    the fraction of that true class misrouted this way.
    """
    errors = frame[~frame[CORRECT_COL]]
    if errors.empty:
        return pd.DataFrame(columns=[ERROR_TYPE_COL, "count", "share_of_errors", "share_of_class"])
    counts = errors.groupby([TRUE_COL, ERROR_TYPE_COL]).size().rename("count").reset_index()
    class_sizes = frame[TRUE_COL].value_counts()
    counts["share_of_errors"] = counts["count"] / counts["count"].sum()
    counts["share_of_class"] = counts["count"] / counts[TRUE_COL].map(class_sizes)
    return (
        counts.sort_values("count", ascending=False)
        .drop(columns=TRUE_COL)
        .reset_index(drop=True)
    )


def error_rate_by(frame: pd.DataFrame, group_col: str, min_support: int = 1) -> pd.DataFrame:
    """Error rate per value of `group_col`, dropping groups under `min_support`.

    Small groups produce wild rates (1 of 2 = 50%), so the support is returned
    alongside and the cut is explicit rather than left to the reader.
    """
    grouped = frame.groupby(group_col, observed=True)[CORRECT_COL]
    out = pd.DataFrame({"n": grouped.size(), "errors": grouped.size() - grouped.sum()})
    out["error_rate"] = out["errors"] / out["n"]
    out = out[out["n"] >= min_support]
    return out.sort_values("error_rate", ascending=False).reset_index()


def error_rate_by_length(frame: pd.DataFrame) -> pd.DataFrame:
    """Error rate across review-length buckets, in length order."""
    bucketed = frame.assign(
        length=pd.cut(frame["n_words"], bins=LENGTH_BINS, labels=LENGTH_LABELS, include_lowest=True)
    )
    out = error_rate_by(bucketed, "length")
    return out.set_index("length").reindex(LENGTH_LABELS).dropna(subset=["n"]).reset_index()


def high_confidence_errors(frame: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Wrong predictions made with probability >= `threshold`, most confident first.

    These are the likeliest label-noise cases: the model is sure, and the
    rating-derived label disagrees.
    """
    if CONFIDENCE_COL not in frame:
        raise ValueError("frame has no confidence column; pass proba to build_error_frame")
    errors = frame[~frame[CORRECT_COL] & (frame[CONFIDENCE_COL] >= threshold)]
    return errors.sort_values(CONFIDENCE_COL, ascending=False).reset_index(drop=True)


def sample_errors(frame: pd.DataFrame, error_type: str, n: int = 10, seed: int = 42) -> pd.DataFrame:
    """A reproducible random sample of one kind of error, for reading by hand."""
    subset = frame[frame[ERROR_TYPE_COL] == error_type]
    return subset.sample(n=min(n, len(subset)), random_state=seed).reset_index(drop=True)


def disagreement(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of which model got each row right: both, only A, only B, neither.

    The two frames must describe the same rows in the same order. Rows only one
    model gets right show what DistilBERT buys over the baseline (or loses).
    """
    if len(frame_a) != len(frame_b) or not (frame_a[TRUE_COL].to_numpy() == frame_b[TRUE_COL].to_numpy()).all():
        raise ValueError("frames are not over the same rows")
    a, b = frame_a[CORRECT_COL].to_numpy(), frame_b[CORRECT_COL].to_numpy()
    return pd.DataFrame(
        {
            "outcome": ["both right", "only A right", "only B right", "both wrong"],
            "count": [(a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()],
        }
    )
