# Milestone 1 — Project Definition

**CS5998 Capstone Project · MSc Data Science & Artificial Intelligence**
**Student:** Imasha Sithumini
**Title:** *Customer Review Analytics for E-Commerce: Sentiment Classification and Unsupervised Issue Discovery from Review Text*
**Repository:** end-to-end NLP pipeline, incremental commits from Week 1

---

## 1. Problem Statement

Online retailers collect customer reviews at a scale no team can read, and then reduce them to a single number: the average star rating. That number reports *how much* customers liked a product but says nothing about *why*. A 3.4-star average is equally consistent with a garment that runs two sizes small, one whose fabric pills after a wash, and one that simply ships late — three problems owned by three different teams, requiring three different fixes. The explanatory content sits entirely in the unstructured review text, which is exactly the part that never gets analysed.

This project treats that gap as a natural-language-processing problem with two coupled tasks over the same corpus of short-form consumer text:

**Task A — Document-level sentiment classification.** Given only the free-text body of a review, predict a three-class sentiment label (`negative` / `neutral` / `positive`) derived from the star rating (1–2 / 3 / 4–5). This is deliberately posed as a *text-only* task: product metadata is withheld from the classifiers so that measured performance reflects what the language itself carries. The core research question is comparative and representational — **how much does a contextual subword representation (fine-tuned DistilBERT) improve over a sparse lexical one (TF-IDF n-grams + linear classifier) on this corpus, and specifically where?** The expected gain concentrates in exactly the linguistic phenomena a bag-of-words cannot encode: scope of negation ("*not* what I expected"), contrastive and concessive constructions ("beautiful colour **but** the fit is impossible"), understatement and hedged criticism ("it's fine, I suppose"), and intensifier-modified polarity. The minority `neutral` class is where these concentrate, and it is where the comparison will be decided.

**Task B — Unsupervised issue discovery.** Induce the recurring themes in the corpus — sizing and fit, fabric and construction quality, colour accuracy, shipping and returns — without a predefined label set, then attach them to the classifier output so each sentiment segment can be explained by the issues driving it, sliced by product department, class and customer age band. This is posed as a second representational comparison: a probabilistic bag-of-words model (LDA) against an embedding-and-cluster model (BERTopic), scored on a common `c_v` coherence metric over an identical tokenised corpus.

The NLP difficulty here is specific to the genre rather than generic. Reviews are **short** — a median of a few dozen tokens — which starves the word co-occurrence statistics LDA depends on. They are **multi-aspect within a single document**: one review routinely praises fabric and condemns fit, so a single document-level label is a lossy summary of mixed underlying opinion. They carry **domain-specific vocabulary** ("boxy", "sheer", "true to size", "petite") whose polarity is context-dependent and absent from general-purpose sentiment lexicons. And the supervision signal is **noisy by construction**: the star rating is a proxy for textual sentiment, not a direct annotation of it, so a subset of reviews is genuinely mislabelled relative to what its text says.

**Scope boundaries.** Data type: English review text, plus structured metadata used only for segmentation and never as a model feature — a choice justified by a measured Cramér's V below 0.04 between every metadata field and the sentiment label. Technique category: NLP — classical ML baseline, transformer fine-tuning, topic modelling. No external LLM APIs are called; every model is trained or fine-tuned by the student on this dataset, keeping the work inside classical ML/DL evaluation rather than LLM prompt engineering.

---

## 2. Data Source(s)

**Primary dataset:** *Women's E-Commerce Clothing Reviews* — Kaggle, public, anonymised real retailer data.
`https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews`

| Property | Detail |
|---|---|
| Size | ~23,486 reviews, 10 columns |
| Text fields | `Review Text` (free text — the modelling input), `Title` |
| Label source | `Rating` (1–5 stars) → 3-class sentiment; `Recommended IND` (0/1) as a secondary target |
| Metadata | `Division Name`, `Department Name`, `Class Name`, `Age`, `Positive Feedback Count` |
| Licence / ethics | Public, research and educational use; no PII — reviewer identity is absent and no re-identification is attempted |

**Why this dataset.** It is real rather than synthetic; it pairs unstructured text with the structured metadata needed for a business-facing breakdown; it carries a built-in evaluation target (rating) *and* genuine room for unsupervised analysis (topics); and at ~23K short documents it is large enough to fine-tune a distilled transformer meaningfully while staying tractable for one student on free-tier GPU within a semester.

**Preparation already fixed.** Exact duplicate rows dropped; rows missing division/department/class dropped; reviews with empty text excluded from modelling. A **70/15/15 train/validation/test split, stratified by `Rating`, seed 42** (n = 15,824 training rows) protects the sparse 1–2 star classes in the held-out sets. The test split is touched once, at the end.

**Known data limitations, stated up front:** a single retailer, a single category (women's clothing), English only, and a static snapshot with no timestamps — so no temporal trend analysis is claimed, and no claim of cross-retailer or cross-language generalisation will be made.

---

## 3. Intended Methods (linked to programme modules)

| Stage | Method | Programme module |
|---|---|---|
| Ingestion & ETL | Load, deduplicate, null handling, reproducible seeded splits, processed-data artefacts | Big Data Analysis |
| Linguistic preprocessing | NLTK: tokenisation, POS-aware WordNet lemmatisation, stopword removal, URL/HTML/digit/punctuation stripping | **Natural Language Processing** |
| Text representation | TF-IDF word n-grams (sparse lexical) vs. subword tokenisation and contextual embeddings (dense) — the axis the whole comparison turns on | **Natural Language Processing** |
| Baseline model | TF-IDF features → Logistic Regression and Linear SVM; imbalance handled via balanced class weights and SMOTE | Advanced Machine Learning |
| Improved model | Fine-tuned **DistilBERT** (`distilbert-base-uncased`) — same target, same splits, same rows | Advanced AI / **Natural Language Processing** |
| Issue discovery | **BERTopic** (sentence embeddings → UMAP → HDBSCAN → stopword-filtered c-TF-IDF) as primary; **LDA** (gensim) as classical comparator | **Natural Language Processing** / Data Mining |
| Evaluation | Accuracy, macro- and per-class P/R/F1, confusion matrix; `c_v` topic coherence; qualitative error analysis of misclassified reviews by linguistic cause | Advanced Machine Learning / **Natural Language Processing** |
| Delivery | Streamlit dashboard — sentiment and dominant issues by department, class and age band | Big Data Analysis |

**Natural Language Processing is the spine module.** Every other module contributes a stage; NLP supplies the object of study. It owns the preprocessing pipeline, both text representations, the topic-modelling methods, `c_v` coherence as an intrinsic NLP evaluation metric, and the error taxonomy (negation scope, mixed-aspect documents, sarcasm) that makes the baseline-vs-transformer comparison explanatory rather than merely numerical.

**A deliberate methodological asymmetry.** The baseline consumes lemmatised, stopword-stripped text; DistilBERT and BERTopic consume the **raw** text. This is intentional rather than an inconsistency: subword tokenisation and contextual embeddings draw signal from precisely the negations, function words and punctuation that preprocessing removes, so feeding them normalised text would handicap them for no benefit. The row set and label set remain identical across all models, so only the *representation* varies — which is the thing under comparison. A matching point applies inside BERTopic: its topic-word extraction is given a stopword-filtered vectoriser, because without one the c-TF-IDF pass yields topics like *"the, it, and, is"* that score well on `c_v` (frequent words co-occur constantly) while being unreadable — the metric rewarding exactly the wrong thing.

**Headline metric: macro-F1, not accuracy.** With roughly 78% of reviews positive, a majority-class predictor scores well on accuracy while being useless. Macro-F1 and the per-class breakdown are the numbers on which the model comparison is decided.

---

## 4. Expected Outputs

1. **A reproducible end-to-end pipeline** — ingest → clean → preprocess → classify → extract topics → visualise — in a public GitHub repository, with every `src/` module covered by a passing `pytest` suite, fixed seeds and pinned dependencies throughout.
2. **A like-for-like model comparison**: TF-IDF + linear baseline against fine-tuned DistilBERT on identical splits, reported as accuracy, macro-F1, per-class P/R/F1 and confusion matrix, with a justified account of *where* the transformer's gain arises rather than a single headline number.
3. **A labelled topic set per product department**, validated quantitatively (`c_v` coherence, LDA vs. BERTopic) and qualitatively (manual spot-checks of representative reviews per topic), with overlapping or incoherent topics reported honestly rather than forced into clean labels.
4. **Linguistic error analysis** over misclassified reviews, categorised by cause — negation scope, mixed-aspect reviews, sarcasm, rating–text mismatch — providing the evidence for the representational argument in Task A.
5. **A Streamlit dashboard** presenting sentiment and dominant issues by department, class and age band: the business-facing deliverable.
6. **Final report** covering methodology, results, comparison, business interpretation, limitations and trade-offs, plus the Reproducibility Checklist (hardware, seeds, splits, hyperparameters, dependencies).

---

## 5. Key Risks and Assumptions

| Risk | Impact | Mitigation |
|---|---|---|
| **Severe class imbalance** — ~78% positive; 1–2 star reviews scarce | Models collapse toward the majority class; `neutral` recall near zero | Stratified splits, balanced class weights, SMOTE on TF-IDF features; macro-F1 as headline metric; per-class reporting always |
| **`neutral` is intrinsically ambiguous** — 3-star text is often genuinely mixed opinion, not neutral sentiment | Caps achievable macro-F1 for *every* model, baseline and transformer alike | Treat as a finding, not a failure: quantify the ceiling through error analysis and report the mixed-aspect nature of 3-star reviews as a substantive result |
| **Label noise** — star rating proxies textual sentiment rather than annotating it | Some "errors" are correct readings of the text against a wrong label | Manually inspect a sample of misclassifications and report the estimated share attributable to label noise rather than model error |
| **Compute for fine-tuning** — measured at 2.9 samples/sec on local CPU (~4.5h for 3 epochs) | Local training impractical | Google Colab T4 (~20 min/run); DistilBERT rather than full BERT; capped epoch budget; hardware recorded for reproducibility |
| **Topic-model subjectivity** — unsupervised topics can be noisy, overlapping, or artefacts of hyperparameters | Weak or unreproducible business conclusions | Two independent models scored on one metric; embeddings cached so sweeps stay affordable; spot-checks against source reviews; uncertain topics documented as uncertain |
| **Short documents weaken LDA** — few tokens per review starve bag-of-words co-occurrence | LDA may underperform for structural reasons | Anticipated and framed as part of the comparison: the short-document handicap is itself a result, not a defect in the experiment |
| **Single-domain scope** | Findings do not transfer across retailers, categories or languages | Claims explicitly scoped to this dataset; generalisation discussed as a limitation, never asserted |

**Assumptions.**
(1) The star rating is a reasonable if imperfect proxy for textual sentiment — the residual mismatch is measured rather than assumed away.
(2) Review text is predominantly English and free of noise beyond what standard preprocessing handles.
(3) Colab free-tier GPU is sufficient to fine-tune a distilled transformer over ~16K training documents within the project timeline.
(4) Product metadata is a segmentation dimension, not a predictive feature — supported by measured association (Cramér's V ≤ 0.035 for every field) rather than asserted.

---

*Prepared for CS5998 Milestone 1 · September 2026*
