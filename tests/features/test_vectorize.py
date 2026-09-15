import pytest

from src.features.vectorize import build_vectorizer, top_features_by_class

CORPUS = [
    "dress fit perfectly love fabric",
    "shirt run small size order",
    "fabric cheap quality poor return",
    "love dress fit true size",
]


def test_build_vectorizer_uses_unigrams_and_bigrams():
    vec = build_vectorizer()
    assert vec.ngram_range == (1, 2)
    assert vec.lowercase is False


def test_vectorizer_produces_matrix_matching_corpus():
    vec = build_vectorizer(min_df=1, max_df=1.0)
    matrix = vec.fit_transform(CORPUS)
    assert matrix.shape[0] == len(CORPUS)
    assert matrix.shape[1] == len(vec.get_feature_names_out())


def test_vectorizer_emits_bigram_features():
    vec = build_vectorizer(min_df=1, max_df=1.0)
    vec.fit(CORPUS)
    features = set(vec.get_feature_names_out())
    assert any(" " in f for f in features)
    assert "run small" in features


def test_max_features_caps_vocabulary():
    vec = build_vectorizer(max_features=5, min_df=1, max_df=1.0)
    vec.fit(CORPUS)
    assert len(vec.get_feature_names_out()) == 5


def test_min_df_drops_rare_terms():
    vec = build_vectorizer(min_df=2, max_df=1.0)
    vec.fit(CORPUS)
    features = set(vec.get_feature_names_out())
    assert "cheap" not in features  # appears in only one document
    assert "fit" in features  # appears in two


def test_top_features_by_class_returns_sorted_terms():
    import numpy as np

    vec = build_vectorizer(min_df=1, max_df=1.0)
    vec.fit(CORPUS)
    n_features = len(vec.get_feature_names_out())
    coefs = np.zeros((2, n_features))
    target_idx = list(vec.get_feature_names_out()).index("fit")
    coefs[0][target_idx] = 5.0

    top = top_features_by_class(vec, coefs, class_index=0, n=3)
    assert top[0][0] == "fit"
    assert len(top) == 3
    assert top[0][1] >= top[1][1]
