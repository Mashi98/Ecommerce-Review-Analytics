"""Draw the system architecture diagram for the Milestone 2 submission.

Regenerate with `python -m scripts.architecture_diagram`; it writes
`reports/figures/architecture.png` (report, slides) and `architecture.svg`.

The diagram shows what is built, with the module that implements each box.
Dashed boxes are planned and not yet implemented.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Fill by where the step runs; border style by status.
LOCAL = "#dbe9f6"
COLAB = "#fde2c4"
DATA = "#e6e6e6"
OUT = "#dff0d8"
INK = "#1f2933"
MUTED = "#52606d"
HELD_OUT = "#c0392b"


def box(ax, x, y, w, h, title, lines=(), module=None, fill=LOCAL, edge=INK, dashed=False, lw=1.3):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=fill, edgecolor=edge, linewidth=lw,
            linestyle=(0, (4, 3)) if dashed else "-",
        )
    )
    top = y + h - 1.6
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    if lines:
        ax.text(x + w / 2, top - 2.3, "\n".join(lines), ha="center", va="top", fontsize=7.4,
                color=INK, linespacing=1.35)
    if module:
        ax.text(x + w / 2, y + 0.9, module, ha="center", va="bottom", fontsize=6.8,
                color=MUTED, family="monospace")


def arrow(ax, start, end, color=INK, dashed=False, rad=0.0, lw=1.3):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11, color=color, linewidth=lw,
            linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
            shrinkA=0, shrinkB=0,
        )
    )


def column_label(ax, x, w, text):
    ax.text(x + w / 2, 62.2, text, ha="center", va="center", fontsize=8.5, color=MUTED,
            fontweight="bold")


def draw() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 9.6))
    ax.set_xlim(0, 120)
    ax.set_ylim(-4, 66)
    ax.axis("off")

    ax.text(60, 65, "E-Commerce Review Analytics — system architecture", ha="center", va="center",
            fontsize=14, fontweight="bold", color=INK)

    c1, c2, c3, c4, c5 = (2, 21), (26, 20), (50, 30), (85, 18), (107, 12)
    for (x, w), label in zip((c1, c2, c3, c4, c5), ("DATA", "FEATURES", "MODELS", "EVALUATION", "OUTPUTS")):
        column_label(ax, x, w, label)

    # ---- Data -------------------------------------------------------------
    box(ax, c1[0], 49, c1[1], 10, "Raw reviews",
        ["Women's E-Commerce", "Clothing Reviews (Kaggle)", "data/raw/reviews.csv"], fill=DATA)
    box(ax, c1[0], 35.5, c1[1], 10, "Ingest & clean",
        ["schema checks, nulls,", "duplicates, blank text"], module="src/data/load.py", fill=LOCAL)
    box(ax, c1[0], 22, c1[1], 10.5, "Stratified split 70 / 15 / 15",
        ["by star rating; seed 42", "target: 1-2 neg, 3 neutral,", "4-5 positive"],
        module="src/data/split.py", fill=LOCAL)
    arrow(ax, (c1[0] + c1[1] / 2, 49), (c1[0] + c1[1] / 2, 45.5))
    arrow(ax, (c1[0] + c1[1] / 2, 35.5), (c1[0] + c1[1] / 2, 32.5))

    sw = 6.4
    for i, (name, edge, lw) in enumerate((("train", INK, 1.3), ("val", INK, 1.3), ("test", HELD_OUT, 2.4))):
        x = c1[0] + i * (sw + 0.9)
        box(ax, x, 11.5, sw, 5.2, name, fill="#f7f7f7", edge=edge, lw=lw)
        arrow(ax, (c1[0] + c1[1] / 2, 22), (x + sw / 2, 16.7), lw=0.9)
    ax.text(c1[0] + c1[1] / 2, 8.6, "test set held out:\nscored once, at the end", ha="center",
            va="top", fontsize=7.4, color=HELD_OUT, fontweight="bold", linespacing=1.3)

    # ---- Features ---------------------------------------------------------
    box(ax, c2[0], 41, c2[1], 12.5, "Text preprocessing",
        ["tokenize, lemmatize,", "remove stopwords", "→ Review Text Clean"], module="src/features/preprocessing.py")
    box(ax, c2[0], 17, c2[1], 12, "Raw review text",
        ["kept as written:", "transformers and sentence", "embeddings use the words", "preprocessing strips"],
        fill=DATA)
    arrow(ax, (c1[0] + c1[1], 29), (c2[0], 47), rad=-0.1)
    arrow(ax, (c1[0] + c1[1], 26), (c2[0], 23), rad=0.1)

    # ---- Models: four lanes (clean-text lanes on top, raw-text lanes below) --
    lane = {"base": (48, 10), "lda": (36.5, 9), "bert": (24.5, 9), "topic": (4, 16)}
    box(ax, c3[0], lane["base"][0], c3[1], lane["base"][1], "Baseline sentiment",
        ["TF-IDF (1-2 grams, 20k features)", "+ logistic regression", "balanced class weights · local CPU"],
        module="src/models/baseline.py")
    box(ax, c3[0], lane["lda"][0], c3[1], lane["lda"][1], "LDA topics (comparison)",
        ["10 topics · gensim", "bag-of-words on clean text"], module="src/models/topics.py")
    box(ax, c3[0], lane["bert"][0], c3[1], lane["bert"][1], "DistilBERT sentiment (improved)",
        ["distilbert-base-uncased, fine-tuned", "class-weighted loss · Colab T4 GPU"],
        module="src/models/transformer.py", fill=COLAB)
    box(ax, c3[0], lane["topic"][0], c3[1], lane["topic"][1], "BERTopic topics (primary)",
        ["MiniLM sentence embeddings → UMAP → HDBSCAN", "→ c-TF-IDF topic words",
         "merge near-duplicates (53 → 45 topics)", "hand-labeled: product / sizing / issue"],
        module="src/models/topics.py")

    # clean text → baseline, LDA ; raw text → DistilBERT, BERTopic
    cx = c2[0] + c2[1]
    arrow(ax, (cx, 49), (c3[0], 53), rad=-0.08)
    arrow(ax, (cx, 45), (c3[0], 41), rad=0.08)
    arrow(ax, (cx, 25), (c3[0], 29), rad=-0.08)
    arrow(ax, (cx, 21), (c3[0], 12), rad=0.1)

    # ---- Evaluation -------------------------------------------------------
    ax.add_patch(FancyBboxPatch((c4[0], 4), c4[1], 54, boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=LOCAL, edgecolor=INK, linewidth=1.3))
    ax.text(c4[0] + c4[1] / 2, 56.4, "Evaluation", ha="center", va="top", fontsize=9, fontweight="bold")
    sections = [
        (50.5, "Classification metrics",
         ["accuracy, macro-F1,", "per-class P / R / F1,", "confusion matrix"]),
        (36.0, "Error analysis",
         ["by error type, star rating,", "review length, department;", "high-confidence errors;",
          "row-level model diff"]),
        (18.5, "Topic coherence",
         ["c_v over a shared corpus", "and dictionary"]),
    ]
    for y_top, heading, body in sections:
        ax.text(c4[0] + c4[1] / 2, y_top, heading, ha="center", va="top", fontsize=8.2,
                fontweight="bold", color=INK)
        ax.text(c4[0] + c4[1] / 2, y_top - 2.3, "\n".join(body), ha="center", va="top",
                fontsize=7.6, color=INK, linespacing=1.4)
    ax.text(c4[0] + c4[1] / 2, 4.9, "src/evaluation/", ha="center", va="bottom", fontsize=6.8,
            color=MUTED, family="monospace")
    mx = c3[0] + c3[1]
    for y_from, y_to in ((53, 53), (41, 41), (29, 29), (12, 12)):
        arrow(ax, (mx, y_from), (c4[0], y_to))

    # ---- Outputs ----------------------------------------------------------
    box(ax, c5[0], 38, c5[1], 20, "Reports",
        ["error-analysis and", "topic reports (md, csv)", "", "results draft", "", "figures"],
        module="reports/ · docs/", fill=OUT)
    box(ax, c5[0], 14, c5[1], 18, "Dashboard",
        ["Streamlit:", "sentiment trends, topics", "by department / class,", "age filter"],
        module="planned", fill="#f7f7f7", dashed=True)
    arrow(ax, (c4[0] + c4[1], 48), (c5[0], 48))
    arrow(ax, (c4[0] + c4[1], 24), (c5[0], 24), dashed=True)

    # held-out test set feeds the final evaluation only
    test_x = c1[0] + 2 * (sw + 0.9) + sw / 2
    # Routed below the model and evaluation boxes so it cannot be mistaken for
    # a box border: down from the test split, across, then up into Evaluation.
    route_y, eval_x = 0.8, c4[0] + c4[1] / 2
    red = dict(color=HELD_OUT, linewidth=1.6, linestyle=(0, (5, 3)))
    ax.plot([test_x, test_x, eval_x], [11.5, route_y, route_y], **red)
    arrow(ax, (eval_x, route_y), (eval_x, 4), color=HELD_OUT, lw=1.6)
    ax.text(46, 2.3, "final test evaluation: both sentiment models, one run",
            ha="center", va="center", fontsize=7.6, color=HELD_OUT, fontweight="bold")

    # ---- Legend -----------------------------------------------------------
    items = [(LOCAL, "-", "runs locally (Python 3.13)"), (COLAB, "-", "runs on Colab T4 GPU"),
             (DATA, "-", "data"), (OUT, "-", "output"), ("#f7f7f7", "--", "planned")]
    x = 30
    for fill, style, label in items:
        ax.add_patch(FancyBboxPatch((x, -3.2), 2.4, 1.8, boxstyle="round,pad=0,rounding_size=0.3",
                                    facecolor=fill, edgecolor=INK, linewidth=1,
                                    linestyle=(0, (3, 2)) if style == "--" else "-"))
        ax.text(x + 3.2, -2.3, label, va="center", fontsize=7.4, color=INK)
        x += 3.2 + len(label) * 0.47 + 3.2
    return fig


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = draw()
    fig.savefig(FIGURES_DIR / "architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_DIR / "architecture.svg", bbox_inches="tight", facecolor="white")
    print(f"wrote {FIGURES_DIR / 'architecture.png'} and architecture.svg")


if __name__ == "__main__":
    main()
