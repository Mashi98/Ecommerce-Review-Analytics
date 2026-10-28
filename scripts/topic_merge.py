"""Merge near-duplicate BERTopic topics and prepare the manual spot-check.

The primary BERTopic fit (min_topic_size 25, auto topic count) finds 53 topics,
several of them near-duplicates: three "jacket" topics, three "xs/xxs" sizing
topics, three "jeans" topics. This refits that model on the cached embeddings,
merges the near-duplicates with `merge_similar_topics`, and reports what the
merge did to coherence and to the department breakdown.

A driver over functions that are unit-tested in `tests/models/test_topics.py`.
The fit is seeded, so topic ids are stable between runs on the same data —
which is what lets `reports/topic_labels.csv` (hand-written labels) key on them.

Writes:
  reports/topic_models_merged.md           before/after, merged groups, topics, departments
  reports/topics_merged_by_department.csv
  reports/topic_samples.csv                sample reviews per merged topic, for the spot-check
"""

from pathlib import Path

import pandas as pd

from scripts.topic_sweep import (
    DEPARTMENT_COL,
    REPORTS_DIR,
    cached_embeddings,
    department_breakdown,
    md_table,
)
from src.models.baseline import load_split
from src.models.topics import (
    MERGE_MAX_TOPIC_SIZE,
    MERGE_MIN_SIMILARITY,
    RANDOM_SEED,
    bertopic_topic_words,
    build_dictionary,
    coherence_cv,
    merge_similar_topics,
    tokenized_corpus,
    topic_summary,
    train_bertopic,
)
from src.models.transformer import TEXT_COL_RAW

SAMPLES_PER_TOPIC = 4
LABELS_PATH = REPORTS_DIR / "topic_labels.csv"


def topic_words(model, topic_id: int, n: int = 4) -> str:
    return ", ".join(w for w, _ in model.get_topic(topic_id)[:n])


def score(model, tokens, dictionary) -> float:
    return coherence_cv(bertopic_topic_words(model), tokens, dictionary)


def outlier_pct(assignments) -> float:
    return round(100 * sum(1 for t in assignments if t == -1) / len(assignments), 1)


def sample_reviews(df: pd.DataFrame, assignments, model, per_topic: int = SAMPLES_PER_TOPIC) -> pd.DataFrame:
    """A few random reviews per topic, with the star rating, for reading by hand."""
    labeled = df.assign(topic=assignments)
    labeled = labeled[labeled["topic"] != -1]
    rows = []
    for topic_id, group in labeled.groupby("topic"):
        picked = group.sample(n=min(per_topic, len(group)), random_state=RANDOM_SEED)
        for _, review in picked.iterrows():
            rows.append(
                {
                    "topic": topic_id,
                    "topic_size": len(group),
                    "top_words": topic_words(model, topic_id, 6),
                    "rating": review["Rating"],
                    "department": review[DEPARTMENT_COL],
                    "review": " ".join(str(review[TEXT_COL_RAW]).split()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_split("train")
    docs = df[TEXT_COL_RAW].fillna("").astype(str).to_list()
    tokens = tokenized_corpus(df)
    dictionary = build_dictionary(tokens)

    model, before_assignments = train_bertopic(docs, embeddings=cached_embeddings(docs))
    before = {
        "topics": len(bertopic_topic_words(model)),
        "cv": score(model, tokens, dictionary),
        "outliers": outlier_pct(before_assignments),
    }
    original_words = {t: topic_words(model, t) for t in model.get_topics() if t != -1}
    original_sizes = {t: model.topic_sizes_[t] for t in original_words}

    groups = merge_similar_topics(model, docs)
    after_assignments = list(model.topics_)
    after = {
        "topics": len(bertopic_topic_words(model)),
        "cv": score(model, tokens, dictionary),
        "outliers": outlier_pct(after_assignments),
    }
    print(f"merged {len(groups)} groups: {before['topics']} -> {after['topics']} topics")

    comparison = pd.DataFrame(
        [
            {"fit": "before merge", "n_topics": before["topics"], "coherence_cv": before["cv"], "outlier_pct": before["outliers"]},
            {"fit": "after merge", "n_topics": after["topics"], "coherence_cv": after["cv"], "outlier_pct": after["outliers"]},
        ]
    )
    merged_rows = [
        {
            "group": i + 1,
            "original_topic": t,
            "size": original_sizes[t],
            "top_words": original_words[t],
        }
        for i, group in enumerate(groups)
        for t in group
    ]
    breakdown = department_breakdown(df, after_assignments, model)
    labels = pd.read_csv(LABELS_PATH) if LABELS_PATH.exists() else None

    topics = topic_summary(model, after_assignments)
    if labels is not None:
        topics = topics.merge(labels, left_on="Topic", right_on="topic", how="left").drop(columns="topic")
        topics = topics[["Topic", "Count", "theme", "Top words"]]
        breakdown = breakdown.merge(labels[["topic", "theme"]], on="topic", how="left")

    lines = [
        "# BERTopic near-duplicate merge",
        "",
        f"Primary fit (min_topic_size 25, auto topic count) on the {len(df)} training reviews.",
        f"Topics merge when every pair in the group has c-TF-IDF cosine >= {MERGE_MIN_SIMILARITY}",
        f"(complete linkage); topics of more than {MERGE_MAX_TOPIC_SIZE} reviews are never merged.",
        "",
        "## Effect on the model",
        "",
        md_table(comparison, col_formats={"outlier_pct": "{:.1f}"}),
        "",
        "## What was merged",
        "",
        md_table(pd.DataFrame(merged_rows)),
        "",
        "## Topics after merge",
        "",
        md_table(topics),
        "",
        "## Topics by department",
        "",
        f"Top 5 topics per department, outlier documents excluded.",
        "",
        md_table(breakdown, float_fmt="{:.1f}"),
        "",
    ]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "topic_models_merged.md").write_text("\n".join(lines), encoding="utf-8")
    breakdown.to_csv(REPORTS_DIR / "topics_merged_by_department.csv", index=False)
    sample_reviews(df, after_assignments, model).to_csv(REPORTS_DIR / "topic_samples.csv", index=False)
    print("wrote reports/topic_models_merged.md, topics_merged_by_department.csv, topic_samples.csv")


if __name__ == "__main__":
    main()
