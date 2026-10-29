# E-Commerce Review Analytics

**CS5998 Capstone Project — Master of Data Science & Artificial Intelligence**
Author: Imasha Sithumini

## Overview

An end-to-end NLP pipeline for e-commerce customer reviews that:
1. Predicts sentiment / recommendation from review text (baseline TF-IDF + classical ML vs. fine-tuned transformer)
2. Extracts recurring themes and issues (sizing, fabric quality, shipping, etc.) via topic modeling
3. Presents findings in a department/category-level analytics view for business interpretation

See [`docs/milestone-1-project-definition.md`](docs/milestone-1-project-definition.md) for the full problem statement, data source, methods, expected outputs, and risks.

## Dataset

[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (Kaggle, public). Not committed to this repo — download the CSV and place it at `data/raw/reviews.csv` before running the pipeline.

## Project structure

```
├── data/
│   ├── raw/            # original dataset (gitignored — see Dataset section)
│   └── processed/      # cleaned/feature-engineered data (gitignored)
├── notebooks/           # exploratory analysis and experiments
├── src/
│   ├── data/            # loading, cleaning, stratified splits
│   ├── features/        # text preprocessing, TF-IDF vectorization
│   ├── models/          # baseline, DistilBERT, LDA and BERTopic
│   └── evaluation/      # metrics, topic coherence, error analysis
├── scripts/             # runnable analyses (error analysis, topic sweep/merge, predict_review, diagram)
├── reports/             # generated reports, tables and topic labels
│   └── figures/         # architecture diagram and other plots
├── docs/                # milestone docs, results draft, reproducibility checklist, Colab guide
├── tests/               # unit tests, mirrors src/ (124 passing)
├── requirements.txt     # unpinned dependency names
└── requirements-lock.txt  # exact versions used for the reported results
```

## Setup

```bash
py -3.13 -m venv .venv        # Python 3.13 recommended — some deps lack wheels on newer versions
.venv\Scripts\activate         # Windows
pip install -r requirements-lock.txt   # exact versions used for the reported results
# pip install -r requirements.txt     # or the latest compatible versions
```

Hardware, seeds, data checksums, splits and hyperparameters are in [`docs/reproducibility.md`](docs/reproducibility.md).

## Testing

Every `src/` module has a matching `tests/` module. A task is only marked Done once its unit tests pass.

```bash
pytest
```

## Status

Milestone 2 (technical checkpoint) work is complete apart from the demo video and submission. Milestone 3 work is the Streamlit dashboard and the final report.

| Component | State |
|---|---|
| Data cleaning, preprocessing, stratified 70/15/15 splits | Done, reproducible (SHA-256 checked) |
| EDA | Done, `notebooks/01_eda.ipynb` |
| Sentiment baseline (TF-IDF + logistic regression) | Done |
| Improved sentiment model (fine-tuned DistilBERT, Colab T4) | Done, `notebooks/02_distilbert_colab.ipynb` |
| Error analysis (both models, validation split) | Done, `reports/error_analysis_*.md` |
| Topic extraction (LDA vs BERTopic, merged and hand-labeled) | Done, `reports/topic_models_merged.md` |
| System architecture diagram | Done, `reports/figures/architecture.png` |
| Reproducibility checklist | Drafted, `docs/reproducibility.md` |
| Streamlit dashboard | Planned |
| Test-set evaluation | Not run yet: the test split is held out for one final evaluation |

### Results so far (validation split, 3,417 reviews)

Sentiment is three-class (1-2 stars negative, 3 neutral, 4-5 positive). About 77% of reviews are positive, so macro-F1 is the headline metric.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + logistic regression | 0.7951 | 0.6245 |
| Fine-tuned DistilBERT | 0.8303 | 0.6823 |

DistilBERT results vary by about 0.006 macro-F1 between runs of the same settings (GPU training is not bit-reproducible), and the neutral class is the weakest for both models. For topics, BERTopic (45 topics after merging near-duplicates) surfaces specific issue topics that LDA's ten broad topics do not. See [`docs/results-draft.md`](docs/results-draft.md) for the full results and error analysis.

### Try it

```bash
python -m scripts.predict_review "I love this top, but it runs small so I sent it back."
```

This trains the baseline and shows the predicted class, probabilities and the words that drove the prediction. DistilBERT is trained on Colab and its weights are not in the repository. Every other result can be regenerated with the commands in [`docs/reproducibility.md`](docs/reproducibility.md).
