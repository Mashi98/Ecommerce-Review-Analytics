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
│   ├── data/            # loading & cleaning
│   ├── features/        # text preprocessing, vectorization
│   ├── models/           # baseline + transformer training
│   └── evaluation/       # metrics, topic coherence, error analysis
├── reports/
│   └── figures/          # generated plots for the report
├── docs/                 # milestone submissions
├── tests/                # unit tests, mirrors src/
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Testing

Every `src/` module has a matching `tests/` module. A task is only marked Done once its unit tests pass.

```bash
pytest
```

## Status

Milestone 1 (Project Definition) drafted. Implementation in progress.
