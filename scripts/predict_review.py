"""Classify review text with the TF-IDF + logistic regression baseline.

Trains the baseline on the training split (a few seconds on CPU), then predicts
sentiment for each review you give it and shows which words drove the answer.
Meant for demos and for poking at failure modes; it never reads the validation
or test splits.

    python -m scripts.predict_review "Love it, but it runs small so I sent it back."
    python -m scripts.predict_review --file my_reviews.txt        # one review per line
    python -m scripts.predict_review                               # type reviews, blank line to quit

The DistilBERT model is not available here: it is trained on Colab and its
weights are not part of the repository.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocessing import preprocess
from src.models.baseline import SENTIMENT_LABELS, load_split, train_baseline

N_TERMS = 5


def read_reviews(path: Path) -> list[str]:
    """One review per line; blank lines are skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def predict_reviews(model, reviews: list[str]) -> pd.DataFrame:
    """Predicted class and class probabilities for each raw review.

    `usable` is False when preprocessing leaves no words (blank text, only
    stopwords or digits): the model then has nothing to go on and its answer is
    just the class prior, so callers should not present it as a prediction.
    """
    cleaned = [preprocess(text) for text in reviews]
    proba = model.predict_proba(cleaned)
    classes = list(model.classes_)
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    out.insert(0, "review", reviews)
    out.insert(1, "predicted", [classes[i] for i in proba.argmax(axis=1)])
    out["usable"] = [bool(text.strip()) for text in cleaned]
    return out


def influential_terms(model, review: str, label: str, n: int = N_TERMS) -> list[tuple[str, float]]:
    """Words in `review` that pushed the model hardest towards `label`.

    The contribution of a term is its TF-IDF weight in this review times the
    logistic-regression coefficient for `label`; only terms the vectorizer
    knows can contribute, and only positive contributions are returned.
    """
    vectorizer, classifier = model.named_steps["tfidf"], model.named_steps["clf"]
    row = vectorizer.transform([preprocess(review)])
    class_index = list(classifier.classes_).index(label)
    contributions = np.asarray(row.multiply(classifier.coef_[class_index]).todense()).ravel()
    names = vectorizer.get_feature_names_out()
    top = [i for i in contributions.argsort()[::-1][:n] if contributions[i] > 0]
    return [(names[i], float(contributions[i])) for i in top]


def format_prediction(row: pd.Series, terms: list[tuple[str, float]]) -> str:
    """A short, readable block for one prediction."""
    if not row["usable"]:
        return "no usable words after preprocessing, so no prediction"
    probs = "  ".join(f"{label} {row[f'p_{label}']:.2f}" for label in SENTIMENT_LABELS)
    lines = [f"{row['predicted']}   ({probs})"]
    if terms:
        lines.append("  driven by: " + ", ".join(f"{term} ({weight:+.2f})" for term, weight in terms))
    return "\n".join(lines)


def show(model, reviews: list[str]) -> None:
    results = predict_reviews(model, reviews)
    for _, row in results.iterrows():
        print(f"\n> {row['review']}")
        terms = influential_terms(model, row["review"], row["predicted"]) if row["usable"] else []
        print(format_prediction(row, terms))


def interactive(model) -> None:
    print("Type a review and press Enter. A blank line quits.")
    while True:
        try:
            text = input("\nreview> ").strip()
        except EOFError:
            break
        if not text:
            break
        show(model, [text])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reviews", nargs="*", help="review text; quote each review")
    parser.add_argument("--file", type=Path, help="text file with one review per line")
    args = parser.parse_args()

    reviews = list(args.reviews)
    if args.file:
        reviews += read_reviews(args.file)

    print("Training the baseline on the training split...", file=sys.stderr)
    model = train_baseline(load_split("train"), model="logreg", imbalance="class_weight")

    if reviews:
        show(model, reviews)
    else:
        interactive(model)


if __name__ == "__main__":
    main()
