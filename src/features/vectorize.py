"""TF-IDF vectorization over preprocessed review text.

Input is expected to be the `Review Text Clean` column produced by
`src.features.preprocessing.preprocess` — already lowercased, stopword-free
and lemmatized — so the vectorizer does no further normalization of its own.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.9


def build_vectorizer(
    max_features: int = MAX_FEATURES,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
) -> TfidfVectorizer:
    """Configured TF-IDF vectorizer for already-preprocessed review text.

    Unigrams + bigrams (bigrams catch phrases like "run small", "true size"),
    dropping terms in fewer than `min_df` documents or more than `max_df` of
    them. `lowercase` is off because `preprocess()` has already lowercased.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        sublinear_tf=True,
    )


def top_features_by_class(vectorizer: TfidfVectorizer, coefficients, class_index: int, n: int = 20) -> list[tuple[str, float]]:
    """The `n` highest-weighted TF-IDF terms for one class of a linear model.

    Used for error analysis and for the report's interpretability section.
    """
    feature_names = vectorizer.get_feature_names_out()
    weights = coefficients[class_index]
    top_idx = weights.argsort()[::-1][:n]
    return [(feature_names[i], float(weights[i])) for i in top_idx]
