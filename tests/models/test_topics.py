"""Unit tests for topic extraction.

LDA and coherence are tested for real on a small synthetic corpus. BERTopic
is not fit here — it downloads a sentence-transformer and takes minutes on
CPU — so only its pure post-processing helpers are covered, with stubs.
"""

import pandas as pd
import pytest

from src.models.topics import (
    bertopic_topic_words,
    build_dictionary,
    coherence_cv,
    lda_topic_words,
    tokenized_corpus,
    train_lda,
)


@pytest.fixture
def token_corpus():
    """Two clearly separable themes: sizing complaints vs. fabric praise."""
    sizing = [["size", "small", "run", "tight", "order", "large"] for _ in range(20)]
    fabric = [["fabric", "soft", "quality", "beautiful", "material", "love"] for _ in range(20)]
    return sizing + fabric


@pytest.fixture
def review_df():
    return pd.DataFrame(
        {
            "Review Text Clean": ["size small run tight", "fabric soft quality", "", None, "   "],
            "Review Text": ["runs small", "lovely fabric", "x", None, "y"],
        }
    )


def test_tokenized_corpus_splits_and_drops_empty(review_df):
    tokens = tokenized_corpus(review_df)
    assert len(tokens) == 2  # blank, None and whitespace-only rows dropped
    assert tokens[0] == ["size", "small", "run", "tight"]


def test_build_dictionary_filters_rare_terms(token_corpus):
    dictionary = build_dictionary(token_corpus, no_below=5, no_above=1.0)
    vocab = set(dictionary.token2id)
    assert "size" in vocab and "fabric" in vocab


def test_build_dictionary_drops_ubiquitous_terms():
    tokens = [["common", f"unique{i}", "other"] for i in range(20)]
    dictionary = build_dictionary(tokens, no_below=1, no_above=0.5)
    # "common" appears in 100% of docs, above the 50% ceiling
    assert "common" not in set(dictionary.token2id)


def test_train_lda_returns_requested_topic_count(token_corpus):
    model, dictionary, corpus = train_lda(token_corpus, num_topics=2, passes=2)
    assert model.num_topics == 2
    assert len(corpus) == len(token_corpus)
    assert len(dictionary) > 0


def test_lda_topic_words_shape(token_corpus):
    model, _, _ = train_lda(token_corpus, num_topics=2, passes=2)
    words = lda_topic_words(model, n_words=4)
    assert len(words) == 2
    assert all(len(topic) == 4 for topic in words)
    assert all(isinstance(w, str) for topic in words for w in topic)


def test_lda_is_deterministic_under_fixed_seed(token_corpus):
    first, _, _ = train_lda(token_corpus, num_topics=2, seed=42, passes=2)
    second, _, _ = train_lda(token_corpus, num_topics=2, seed=42, passes=2)
    assert lda_topic_words(first) == lda_topic_words(second)


def test_lda_separates_two_planted_themes(token_corpus):
    """With two obviously distinct themes, the topics should not be identical."""
    model, _, _ = train_lda(token_corpus, num_topics=2, passes=20)
    topic_a, topic_b = (set(t) for t in lda_topic_words(model, n_words=3))
    assert topic_a != topic_b


def test_coherence_cv_returns_finite_score(token_corpus):
    dictionary = build_dictionary(token_corpus, no_below=1, no_above=1.0)
    score = coherence_cv([["size", "small", "run"], ["fabric", "soft", "quality"]], token_corpus, dictionary)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_coherence_cv_drops_out_of_vocabulary_words(token_corpus):
    """Unknown words must be filtered, not crash the scorer."""
    dictionary = build_dictionary(token_corpus, no_below=1, no_above=1.0)
    score = coherence_cv([["size", "small", "zzzznotaword"], ["fabric", "soft"]], token_corpus, dictionary)
    assert isinstance(score, float)


def test_coherence_cv_raises_when_nothing_survives(token_corpus):
    dictionary = build_dictionary(token_corpus, no_below=1, no_above=1.0)
    with pytest.raises(ValueError, match="No topic survived"):
        coherence_cv([["zzzz", "yyyy"], ["xxxx"]], token_corpus, dictionary)


def test_bertopic_topic_words_excludes_outlier_topic():
    """Topic -1 is BERTopic's unclustered catch-all and must be dropped."""

    class StubBERTopic:
        _topics = {
            -1: [("junk", 0.1), ("noise", 0.05)],
            0: [("size", 0.9), ("small", 0.8), ("tight", 0.7)],
            1: [("fabric", 0.9), ("soft", 0.8), ("quality", 0.7)],
        }

        def get_topics(self):
            return self._topics

        def get_topic(self, topic_id):
            return self._topics[topic_id]

    words = bertopic_topic_words(StubBERTopic(), n_words=2)
    assert words == [["size", "small"], ["fabric", "soft"]]
    assert not any("junk" in topic for topic in words)


def test_topic_vectorizer_strips_stopwords():
    """Without this, BERTopic topics come out as "the, it, and" — high c_v,
    zero interpretability."""
    from src.models.topics import build_topic_vectorizer

    vectorizer = build_topic_vectorizer(min_df=1)
    vectorizer.fit(["the dress is beautiful and the fabric is soft"])
    features = set(vectorizer.get_feature_names_out())
    assert not ({"the", "is", "and"} & features)
    assert {"dress", "fabric", "soft"} <= features
