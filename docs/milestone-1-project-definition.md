# Milestone 1: Project Definition
**CS5998 — Capstone Project | Master of Data Science & Artificial Intelligence**
**Title:** Customer Review Analytics Pipeline for E-Commerce — Sentiment Classification and Issue/Topic Extraction

---

## 1. Problem Statement

Online retailers accumulate large volumes of unstructured customer reviews that contain valuable signals about product quality, fit, and service issues, but this feedback is rarely analyzed systematically beyond aggregate star ratings. Manually reading reviews does not scale, and raw star ratings alone do not explain *why* customers are satisfied or dissatisfied.

This project builds an end-to-end NLP pipeline that (a) predicts customer sentiment/recommendation from review text, and (b) automatically extracts the recurring themes and issues (e.g. sizing, fabric quality, shipping, durability) driving those sentiments, broken down by product department and category. The goal is to demonstrate a realistic decision-support pipeline a retail analytics team could use to prioritize product and service improvements.

**Problem type:** Text classification (sentiment / recommendation prediction) combined with comparative model evaluation and unsupervised topic extraction.

**Bounded scope choices:**
- Data type: Text (review content) + structured metadata (rating, department, class, customer age)
- Technique category: NLP (classical ML baseline, transformer-based fine-tuning, topic modeling)
- System context: End-to-end pipeline with a lightweight analytics dashboard for business interpretation

No external LLM APIs are used in this project — all models are trained/fine-tuned locally on the dataset, keeping the project fully within classical ML/DL evaluation (not subject to the LLM-wrapper restrictions).

---

## 2. Data Source(s)

**Primary dataset:** Women's E-Commerce Clothing Reviews (Kaggle, public domain, anonymized real retailer data)
- ~23,486 rows, 10 columns: Review Text, Title, Rating (1–5), Recommended IND (0/1), Positive Feedback Count, Division Name, Department Name, Class Name, Age
- Source: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
- License: publicly available for research/educational use; no PII (reviewer identity is not included)

**Why this dataset:** it is real, moderately sized (fits a single-semester individual project without heavy compute needs), has both unstructured text and structured metadata (enabling richer analysis and business segmentation), and has a natural evaluation target (Rating / Recommended IND) plus room for unsupervised analysis (topics per department).

**Data risk:** dataset is single-retailer, single-category (women's clothing) and English-only — findings will not generalize across retailers or languages; this will be stated explicitly as a limitation.

---

## 3. Intended Methods (linked to program modules)

| Stage | Method | Linked module |
|---|---|---|
| Preprocessing & pipeline design | Tokenization, stopword removal, lemmatization, ingestion/ETL structuring, handling class imbalance (few 1–2 star reviews) | Big Data Analysis |
| Baseline model | TF-IDF vectorization + Logistic Regression / Linear SVM for sentiment & recommend prediction | Advanced Machine Learning |
| Improved model | Fine-tuned transformer (DistilBERT) for the same classification task, compared against baseline | Advanced AI |
| Topic/issue extraction | LDA and/or BERTopic to surface recurring themes per department, evaluated with topic coherence (c_v) | Data Mining |
| Evaluation | Accuracy, Precision/Recall/F1, confusion matrix, per-class performance, error analysis, topic coherence | Advanced Machine Learning |
| Presentation | Aggregated analytics dashboard summarizing sentiment and topics by department/class for business interpretation | Big Data Analysis |

**Comparative evaluation:** baseline (TF-IDF + classical ML) vs. improved (fine-tuned transformer) satisfies the guide's requirement for an experimental comparison with justification, not just a single model.

---

## 4. Expected Outputs

1. A reproducible, end-to-end pipeline (ingest → clean → classify → extract topics → visualize), with public GitHub repo and meaningful incremental commits from week 1.
2. A trained baseline model and an improved (fine-tuned transformer) model, with a documented, justified comparison of their performance.
3. A set of extracted, labeled topics/issues per product department, validated qualitatively (spot-checked against review text) and quantitatively (coherence score).
4. A lightweight dashboard/report visualizing sentiment trends and top issues by department/class — the business-facing deliverable.
5. Final report documenting methodology, results, error analysis, limitations, and trade-offs, plus the Reproducibility Checklist (Appendix A of the guide).

---

## 5. Key Risks and Assumptions

**Risks:**
- **Class imbalance:** most reviews are positive (4–5 star); minority classes (1–2 star) may be under-represented. *Mitigation:* class weighting, stratified sampling, and/or oversampling (e.g. SMOTE on TF-IDF features); report per-class metrics, not just overall accuracy.
- **Compute constraints:** fine-tuning a transformer requires GPU access. *Mitigation:* use Google Colab (free/T4 GPU tier) and a lightweight model (DistilBERT) with a capped training budget.
- **Topic model subjectivity:** unsupervised topics can be noisy or hard to label. *Mitigation:* validate topics against coherence scores and manual spot-checks of sample reviews per topic; document uncertain/overlapping topics rather than forcing clean labels.
- **Single-domain generalizability:** results are specific to one retailer/category and may not transfer. *Mitigation:* explicitly scope claims to this dataset in the final report; discuss generalizability as a limitation, not a claimed strength.

**Assumptions:**
- The Kaggle dataset's ratings/recommend flags are a reasonable (if imperfect) proxy for ground-truth sentiment.
- Review text is predominantly English and free of significant noise beyond what standard NLP preprocessing handles.
- Available compute (Colab free tier) is sufficient for fine-tuning a distilled transformer model on ~23K short text samples within the project timeline.

---

*Draft prepared for CS5998 Milestone 1 submission.*
