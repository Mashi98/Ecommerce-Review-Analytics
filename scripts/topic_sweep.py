"""Tuning sweep for the two topic models, plus per-department topic labeling.

Settles the open decision in `Brain/project-architecture.md` section 5 — which
BERTopic configuration to run as the primary model — with coherence numbers
rather than judgment, and produces the topic comparison Milestone 2 needs.

A driver script, not library code: it only composes functions from
`src.models.topics`, which are unit-tested in `tests/models/test_topics.py`.

Embedding dominates the runtime on CPU, so it is computed once and reused
across every BERTopic configuration (this is what commit 43e1195 split
`embed_documents` out for). Coherence is scored over one shared tokenized
corpus and dictionary, so the numbers compare topics rather than tokenization.

Writes:
  reports/topic_model_sweep.csv   the raw sweep table
  reports/topic_models.md         sweep + labeled topics + department breakdown
"""

from pathlib import Path

import pandas as pd

from src.models.baseline import load_split
from src.models.topics import (
    RANDOM_SEED,
    TOP_N_WORDS,
    build_dictionary,
    coherence_cv,
    embed_documents,
    lda_topic_words,
    tokenized_corpus,
    train_bertopic,
    train_lda,
)
from src.models.transformer import TEXT_COL_RAW

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

LDA_TOPIC_COUNTS = [10, 15, 20]

# (min_topic_size, nr_topics). nr_topics=None lets HDBSCAN choose the count;
# the current default is (50, 10), the 2026-09-19 proposal is (25, None).
BERTOPIC_CONFIGS = [(50, 10), (50, None), (25, 10), (25, None)]

DEPARTMENT_COL = "Department Name"
TOP_TOPICS_PER_DEPARTMENT = 5


def md_table(
    df: pd.DataFrame,
    float_fmt: str = "{:.4f}",
    col_formats: dict[str, str] | None = None,
) -> str:
    """Markdown table from a DataFrame.

    `DataFrame.to_markdown` needs `tabulate`, which is not a project dependency
    and is not worth adding for report formatting alone.

    `col_formats` overrides `float_fmt` per column — coherence wants four
    decimals in the same table where a percentage wants one.
    """
    col_formats = col_formats or {}

    def cell(value, column: str) -> str:
        # Missing values must be tested before the float branch: NaN is itself
        # a float, so checking the other way round renders it as "nan".
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, float):
            if pd.isna(value):
                return ""
            return col_formats.get(column, float_fmt).format(value)
        return str(value)

    header = [str(c) for c in df.columns]
    rows = [
        [cell(v, str(c)) for v, c in zip(row, df.columns)]
        for row in df.itertuples(index=False)
    ]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join(out)


def run_sweep(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit every configuration, score all of them on the same c_v setup."""
    tokens = tokenized_corpus(df)
    dictionary = build_dictionary(tokens)
    docs = df[TEXT_COL_RAW].fillna("").astype(str).to_list()

    print(f"corpus: {len(df)} reviews | {len(tokens)} non-empty token lists")
    print(f"embedding {len(docs)} documents once, reused across configs...")
    embeddings = embed_documents(docs)

    rows = []
    fitted = {}

    for k in LDA_TOPIC_COUNTS:
        print(f"\n--- LDA k={k} ---")
        model, _, _ = train_lda(tokens, num_topics=k, seed=RANDOM_SEED)
        words = lda_topic_words(model, n_words=TOP_N_WORDS)
        score = coherence_cv(words, tokens, dictionary)
        config = f"num_topics={k}"
        rows.append(
            {
                "model": "LDA",
                "config": config,
                "n_topics": len(words),
                "coherence_cv": score,
            }
        )
        # Keyed by the same config string the summary carries, so the report
        # can look a fitted model up straight from the winning row.
        fitted[config] = {"words": words, "model": model, "assignments": None}
        print(f"LDA {config}: c_v {score:.4f} over {len(words)} topics")

    for min_size, nr_topics in BERTOPIC_CONFIGS:
        print(f"\n--- BERTopic min_topic_size={min_size} nr_topics={nr_topics} ---")
        model, assignments = train_bertopic(
            docs,
            num_topics=nr_topics,
            seed=RANDOM_SEED,
            min_topic_size=min_size,
            embeddings=embeddings,
        )
        # Imported late: bertopic_topic_words needs the fitted model only.
        from src.models.topics import bertopic_topic_words

        words = bertopic_topic_words(model, n_words=TOP_N_WORDS)
        score = coherence_cv(words, tokens, dictionary)
        outliers = sum(1 for t in assignments if t == -1)
        config = f"min_topic_size={min_size}, nr_topics={nr_topics}"
        rows.append(
            {
                "model": "BERTopic",
                "config": config,
                "n_topics": len(words),
                "coherence_cv": score,
                "outlier_docs": outliers,
                "outlier_pct": round(100 * outliers / len(assignments), 1),
            }
        )
        fitted[config] = {"words": words, "model": model, "assignments": assignments}
        print(
            f"BERTopic {config}: c_v {score:.4f} over {len(words)} topics, "
            f"{outliers} outlier docs ({100 * outliers / len(assignments):.1f}%)"
        )

    summary = pd.DataFrame(rows).sort_values("coherence_cv", ascending=False)
    if "outlier_docs" in summary:
        # Nullable integer: the LDA rows have no outlier count, and a plain int
        # column cannot carry that gap without turning the whole column float
        # and printing counts as "224.0000".
        summary["outlier_docs"] = summary["outlier_docs"].astype("Int64")
    return summary.reset_index(drop=True), fitted


def department_breakdown(df: pd.DataFrame, assignments: list[int], model) -> pd.DataFrame:
    """Top topics per department, with readable labels.

    Outlier documents (topic -1) are excluded — they are BERTopic's "could not
    cluster" bucket, not a theme anyone can act on.
    """
    labeled = df.assign(topic=assignments)
    labeled = labeled[labeled["topic"] != -1]

    def label(topic_id: int) -> str:
        return ", ".join(w for w, _ in model.get_topic(topic_id)[:5])

    rows = []
    for dept, group in labeled.groupby(DEPARTMENT_COL):
        counts = group["topic"].value_counts().head(TOP_TOPICS_PER_DEPARTMENT)
        for topic_id, count in counts.items():
            rows.append(
                {
                    "department": dept,
                    "topic": topic_id,
                    "reviews": int(count),
                    "share_of_dept": round(100 * count / len(group), 1),
                    "top_words": label(topic_id),
                }
            )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, fitted: dict, df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "topic_model_sweep.csv", index=False)

    best_bertopic = summary[summary["model"] == "BERTopic"].iloc[0]
    best_lda = summary[summary["model"] == "LDA"].iloc[0]

    bt = fitted[best_bertopic["config"]]
    lda = fitted[best_lda["config"]]
    breakdown = department_breakdown(df, bt["assignments"], bt["model"])
    breakdown.to_csv(REPORTS_DIR / "topics_by_department.csv", index=False)

    lines = [
        "# Topic Model Comparison and Tuning Sweep",
        "",
        f"Fit on the {len(df)} reviews of the training split. All configurations",
        "scored on c_v coherence over the same tokenized corpus and dictionary.",
        "",
        "## Sweep results",
        "",
        md_table(summary, col_formats={"outlier_pct": "{:.1f}"}),
        "",
        f"Best BERTopic: **{best_bertopic['config']}** (c_v {best_bertopic['coherence_cv']:.4f})",
        f"Best LDA: **{best_lda['config']}** (c_v {best_lda['coherence_cv']:.4f})",
        "",
        f"## BERTopic topics — {best_bertopic['config']}",
        "",
    ]

    from src.models.topics import topic_summary

    lines.append(md_table(topic_summary(bt["model"], bt["assignments"])))
    lines += ["", f"## LDA topics — {best_lda['config']}", ""]
    for i, words in enumerate(lda["words"]):
        lines.append(f"{i:>2}. {', '.join(words)}")

    lines += [
        "",
        "## Topics by department",
        "",
        f"Top {TOP_TOPICS_PER_DEPARTMENT} topics per department, outlier documents excluded.",
        "",
        md_table(breakdown, float_fmt="{:.1f}"),
        "",
    ]

    (REPORTS_DIR / "topic_models.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {REPORTS_DIR / 'topic_models.md'}")
    print(f"wrote {REPORTS_DIR / 'topic_model_sweep.csv'}")
    print(f"wrote {REPORTS_DIR / 'topics_by_department.csv'}")


if __name__ == "__main__":
    reviews = load_split("train")
    sweep, models = run_sweep(reviews)

    print("\n=== Sweep summary (c_v, higher is better) ===")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    write_report(sweep, models, reviews)
