"""Topic / issue extraction from review text.

Two models, scored on the same metric so they can be compared:
  - **LDA** (gensim) — the classical comparison baseline
  - **BERTopic** — the primary model (see Brain/project-architecture.md)

Both are evaluated with c_v topic coherence over the *same* tokenized corpus
and dictionary, so the score reflects the topics rather than differences in
tokenization.

LDA consumes the preprocessed `Review Text Clean` tokens (bag-of-words needs
the stopword/lemma normalization). BERTopic embeds the **raw** `Review Text`
for the same reason DistilBERT does — its sentence embeddings use the words
`preprocess()` strips.

BERTopic's *topic words*, however, come from a c-TF-IDF pass that must strip
stopwords itself. Without that it produces topics like "the, it, and, is" that
score high on c_v (frequent words co-occur constantly) while being useless to
read — the metric rewards exactly the wrong thing. `build_topic_vectorizer()`
supplies a stopword-filtered CountVectorizer to prevent this.
"""

from pathlib import Path

import pandas as pd

from src.models.baseline import TEXT_COL, load_split
from src.models.transformer import TEXT_COL_RAW

RANDOM_SEED = 42
NUM_TOPICS = 10
TOP_N_WORDS = 10

# Dictionary filtering: drop terms in <5 documents or >50% of them.
NO_BELOW = 5
NO_ABOVE = 0.5

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def tokenized_corpus(df: pd.DataFrame, text_col: str = TEXT_COL) -> list[list[str]]:
    """Token lists from the preprocessed text column, dropping empty docs."""
    texts = df[text_col].fillna("").astype(str)
    return [tokens for tokens in (t.split() for t in texts) if tokens]


def build_dictionary(tokens: list[list[str]], no_below: int = NO_BELOW, no_above: float = NO_ABOVE):
    """Gensim Dictionary with extreme-frequency terms filtered out."""
    from gensim.corpora import Dictionary

    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary


def train_lda(
    tokens: list[list[str]],
    num_topics: int = NUM_TOPICS,
    seed: int = RANDOM_SEED,
    passes: int = 10,
):
    """Fit an LDA model. Returns (model, dictionary, bag-of-words corpus)."""
    from gensim.models import LdaModel

    dictionary = build_dictionary(tokens)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=passes,
        alpha="auto",
    )
    return model, dictionary, corpus


def lda_topic_words(model, n_words: int = TOP_N_WORDS) -> list[list[str]]:
    """Top-`n_words` terms per LDA topic, as plain word lists."""
    return [
        [word for word, _ in model.show_topic(topic_id, topn=n_words)]
        for topic_id in range(model.num_topics)
    ]


def coherence_cv(topic_words: list[list[str]], tokens: list[list[str]], dictionary) -> float:
    """c_v coherence for a set of topics, scored against a reference corpus.

    Topic words absent from the dictionary are dropped, and topics left with
    fewer than two words are skipped — gensim errors on them otherwise.
    """
    from gensim.models import CoherenceModel

    vocab = set(dictionary.token2id)
    filtered = [[w for w in topic if w in vocab] for topic in topic_words]
    filtered = [topic for topic in filtered if len(topic) >= 2]
    if not filtered:
        raise ValueError("No topic survived dictionary filtering — cannot score coherence")

    return float(
        CoherenceModel(
            topics=filtered, texts=tokens, dictionary=dictionary, coherence="c_v"
        ).get_coherence()
    )


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_topic_vectorizer(min_df: int = 5, ngram_range: tuple[int, int] = (1, 2)):
    """CountVectorizer for BERTopic's c-TF-IDF topic representation.

    Strips English stopwords so topic labels are readable content words rather
    than "the, it, and". This affects only how topics are *described* — the
    document embeddings and clustering still see the full raw text.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(stop_words="english", min_df=min_df, ngram_range=ngram_range)


def train_bertopic(
    docs: list[str],
    num_topics: int | None = None,
    seed: int = RANDOM_SEED,
    embedding_model: str = EMBEDDING_MODEL,
    min_topic_size: int = 50,
    vectorizer_model=None,
):
    """Fit BERTopic over raw review text.

    UMAP is seeded explicitly — without `random_state` BERTopic is
    non-deterministic between runs, which would break the reproducibility
    checklist. `num_topics` caps the count via topic reduction; leave it None
    to let HDBSCAN decide.
    """
    from bertopic import BERTopic
    from umap import UMAP

    vectorizer_model = vectorizer_model if vectorizer_model is not None else build_topic_vectorizer()
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics=num_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = model.fit_transform(docs)
    return model, topics


def bertopic_topic_words(model, n_words: int = TOP_N_WORDS) -> list[list[str]]:
    """Top-`n_words` terms per BERTopic topic, excluding the -1 outlier topic.

    Topic -1 is BERTopic's catch-all for documents it could not cluster; it is
    not a real topic and must be excluded before scoring coherence.
    """
    return [
        [word for word, _ in model.get_topic(topic_id)[:n_words]]
        for topic_id in sorted(model.get_topics())
        if topic_id != -1
    ]


def topic_summary(model, topics: list[int], n_words: int = TOP_N_WORDS) -> pd.DataFrame:
    """Per-topic size and top terms, largest first (outlier topic excluded)."""
    info = model.get_topic_info()
    info = info[info["Topic"] != -1].copy()
    info["Top words"] = [
        ", ".join(w for w, _ in model.get_topic(t)[:n_words]) for t in info["Topic"]
    ]
    return info[["Topic", "Count", "Top words"]].reset_index(drop=True)


def compare_topic_models(
    df: pd.DataFrame,
    num_topics: int = NUM_TOPICS,
    seed: int = RANDOM_SEED,
    n_words: int = TOP_N_WORDS,
) -> tuple[pd.DataFrame, dict]:
    """Fit LDA and BERTopic on the same reviews and score both on c_v coherence.

    Returns a summary DataFrame and a dict of the fitted models/topic words.
    """
    tokens = tokenized_corpus(df)
    dictionary = build_dictionary(tokens)

    lda, _, _ = train_lda(tokens, num_topics=num_topics, seed=seed)
    lda_words = lda_topic_words(lda, n_words=n_words)

    docs = df[TEXT_COL_RAW].fillna("").astype(str).to_list()
    bertopic_model, bertopic_assignments = train_bertopic(docs, num_topics=num_topics, seed=seed)
    bertopic_words = bertopic_topic_words(bertopic_model, n_words=n_words)

    summary = pd.DataFrame(
        [
            {
                "model": "LDA",
                "n_topics": len(lda_words),
                "coherence_cv": coherence_cv(lda_words, tokens, dictionary),
            },
            {
                "model": "BERTopic",
                "n_topics": len(bertopic_words),
                "coherence_cv": coherence_cv(bertopic_words, tokens, dictionary),
            },
        ]
    ).sort_values("coherence_cv", ascending=False).reset_index(drop=True)

    return summary, {
        "lda": lda,
        "lda_words": lda_words,
        "bertopic": bertopic_model,
        "bertopic_words": bertopic_words,
        "bertopic_assignments": bertopic_assignments,
        "tokens": tokens,
        "dictionary": dictionary,
    }


if __name__ == "__main__":
    reviews = load_split("train")
    print(f"fitting topic models on {len(reviews)} training reviews\n")

    summary_df, fitted = compare_topic_models(reviews)
    print("\nCoherence comparison (c_v, higher is better):")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nLDA topics:")
    for i, words in enumerate(fitted["lda_words"]):
        print(f"  {i:>2}: {', '.join(words)}")

    print("\nBERTopic topics:")
    print(topic_summary(fitted["bertopic"], fitted["bertopic_assignments"]).to_string(index=False))
