# Results draft (Milestone 2 / final report)

Working draft of the results and error-analysis sections. Every number is from a file in `reports/`. Validation results (1.1) and the single test-set evaluation (1.2) both use the final DistilBERT run, whose weights are saved. Section 2 (error analysis) is on validation predictions from the earlier Milestone 2 run (see its note).

## 1. Sentiment classification

**Task.** Three-class sentiment derived from the star rating: 1-2 negative, 3 neutral, 4-5 positive. The splits are stratified 70/15/15; reviews with no text are dropped, leaving 15,824 training, 3,417 validation and 3,386 test reviews. About 77% of validation reviews are positive, so accuracy flatters any model that favours the majority class and **macro-F1 is the headline metric**.

**Models.**

| | Baseline | Improved |
|---|---|---|
| Model | TF-IDF + logistic regression | `distilbert-base-uncased`, fine-tuned |
| Input | lemmatized, stopword-free text | raw review text (max 256 tokens) |
| Imbalance | balanced class weights | inverse-frequency class weights |
| Features / settings | 20,000 features, 1-2 grams, min_df 2 | lr 2e-5, batch 16, 3 epochs, weight decay 0.01, 500 warmup steps, fp16, seed 42 |
| Hardware | local CPU | Colab T4 GPU |

DistilBERT is given raw text because its subword tokenizer uses the words that lemmatization and stopword removal strip. The checkpoint with the best validation macro-F1 is kept.

### 1.1 Validation results

| | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + LogReg | 0.7951 | 0.6245 | 0.8083 |
| DistilBERT (final run) | 0.8288 | **0.6785** | 0.8411 |

The final run (585 validation errors) replaces the Milestone 2 run (0.8303 / 0.6823, 580 errors) as the reported model, because its weights were saved and it can be scored on the test set. The two runs differ by 0.004 macro-F1, within the run-to-run noise noted below.

| Class | Baseline P / R / F1 | DistilBERT P / R / F1 | Support |
|---|---|---|---|
| negative | 0.514 / 0.617 / 0.561 | 0.605 / 0.682 / 0.641 | 358 |
| neutral | 0.359 / 0.467 / 0.406 | 0.403 / 0.549 / 0.465 | 426 |
| positive | 0.945 / 0.872 / 0.907 | 0.967 / 0.894 / 0.929 | 2,633 |

DistilBERT improves every class, and macro-F1 by 0.054. Most of the gain is on the minority classes (negative +0.080 F1, neutral +0.059), which is what macro-F1 rewards. The neutral class stays weak for both models.

### 1.2 Test results (evaluated once)

The 3,386-review test split was scored once by each model, after all model and setting choices were made on validation. DistilBERT predictions come from the final Colab run (weights, run info and predictions saved to Drive); the baseline was retrained locally and first checked to reproduce its validation macro-F1 of 0.6245. Every prediction row was matched to the local split by a digest of the review text. Source: `reports/test_results.md`.

| | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + LogReg | 0.7930 | 0.6182 | 0.8059 |
| DistilBERT | 0.8269 | **0.6670** | 0.8379 |

| Class | Baseline P / R / F1 | DistilBERT P / R / F1 | Support |
|---|---|---|---|
| negative | 0.491 / 0.598 / 0.539 | 0.585 / 0.668 / 0.624 | 356 |
| neutral | 0.364 / 0.467 / 0.409 | 0.397 / 0.512 / 0.447 | 420 |
| positive | 0.943 / 0.872 / 0.906 | 0.963 / 0.899 / 0.930 | 2,610 |

**Is the gain real?** A paired bootstrap over test reviews (2,000 resamples) gives a DistilBERT minus baseline macro-F1 of **+0.0488, 95% interval [+0.0249, +0.0741]**. DistilBERT beat the baseline in every resample. The interval excludes zero, but its lower end (+0.025) is about half the point estimate, so the size of the gain is uncertain even though its sign is not.

**Validation to test.** Macro-F1 falls by 0.006 for the baseline (0.6245 to 0.6182) and 0.012 for DistilBERT (0.6785 to 0.6670). The small drops fit the expectation that selection on validation made those numbers slightly optimistic; there is no sign of serious overfitting to it. The test numbers are the ones to quote.

**By class.** The pattern matches validation. DistilBERT's gains are largest on negative (F1 +0.085) and positive (+0.024); neutral improves least (+0.038) and remains the weakest class for both models. Looking at the confusion matrices, DistilBERT cuts positive reviews wrongly called negative from 90 to 34 and negative reviews wrongly called positive from 45 to 20, but neutral errors barely move (negative to neutral 98 in both; positive to neutral 244 to 229).

> Note for the reproducibility section: DistilBERT training on a GPU is not bit-reproducible even with a fixed seed. Three runs with identical settings gave macro-F1 0.676 and 0.682 (Milestone 2) and 0.6785 (final), about 0.006 of run-to-run noise. The Milestone 2 weights were lost; the final run's are saved.

## 2. Error analysis

Computed on the validation split only (`reports/error_analysis_baseline.md`, `reports/error_analysis_distilbert.md`). The DistilBERT figures in this section are from the Milestone 2 run (580 errors, `reports/val_preds_distilbert.csv`), not the final run (585 errors); the two differ by five errors and the patterns below are not expected to change. Section 2.6 repeats the analysis on the test set.

### 2.1 Where the models disagree

| Outcome | Reviews |
|---|---|
| Both right | 2,490 |
| Only DistilBERT right | 347 |
| Only baseline right | 227 |
| Both wrong | 353 |

DistilBERT fixes 347 reviews and breaks 227 the baseline got right, a net gain of 120 (700 → 580 errors). The 353 reviews both models get wrong are about half of the baseline's errors, and are the best candidates for label noise or genuinely ambiguous text.

### 2.2 What kind of errors

| Error | Baseline | DistilBERT |
|---|---|---|
| positive → neutral | 254 | 248 |
| neutral → negative | 127 | 118 |
| negative → neutral | 102 | 104 |
| neutral → positive | 100 | 68 |
| positive → negative | 82 | 29 |
| negative → positive | 35 | 13 |

DistilBERT's gain is almost entirely in the opposite-polarity mistakes: positive → negative falls by about two thirds and negative → positive by about 60%. The mistakes that involve the neutral class barely move. Positive → neutral is the largest error type for both models, 36% of baseline errors and 43% of DistilBERT's.

### 2.3 Error rate by star rating

| Rating | Baseline | DistilBERT |
|---|---|---|
| 1 | 30.6% | 19.4% |
| 2 | 42.3% | 39.7% |
| 3 | 53.3% | 43.7% |
| 4 | 29.6% | 28.7% |
| 5 | 6.2% | 3.4% |

The error rate peaks in the middle of the scale. A 3-star review is a coin flip for the baseline, and 4-star reviews are misclassified almost a third of the time by both models. Reading these reviews shows why: a typical 4-star error reads "I really like this tunic, it's gorgeous, but it runs small... I sent it back." The text carries a complaint and a recommendation at once, and the rating-derived label picks one side. The cross-polarity gain concentrates at the ends of the scale (1-star errors fall 11 points), while the 4-star rate is nearly unchanged.

### 2.4 Label noise

Sentiment is derived from the rating, so some "errors" are label disagreements rather than model failures. Among the baseline's 29 errors made with probability of 0.8 or more, many are 3-star reviews that read as praise ("Love the feel and comfort") or 4-star reviews that read as a return. This argues for reporting the neutral-class ceiling as a property of the labels, not only of the models, and it is the strongest reason not to over-interpret small macro-F1 differences.

### 2.5 Review length and product group

For the baseline, the error rate rises with review length, from 15% at 20 words or fewer to 27% above 100 words; longer reviews mix more opinions. The two departments with the most reviews differ little (Dresses 21.6%, Tops 20.0% for the baseline), so department is not a major driver of difficulty. The class with the highest error rate, Swim (35%), has only 57 validation reviews and should not be over-read.

### 2.6 Test-set error analysis

The same breakdown on the test split, for the final DistilBERT run and the baseline (`reports/error_analysis_distilbert_test.md`, `reports/error_analysis_baseline_test.md`). This is descriptive only: nothing was refit or tuned on it. The validation-set patterns in 2.1 to 2.5 hold.

| Outcome (test) | Reviews |
|---|---|
| Both right | 2,478 |
| Only DistilBERT right | 322 |
| Only baseline right | 207 |
| Both wrong | 379 |

DistilBERT makes 586 errors against 701 for the baseline (net +115). The 379 shared errors are 54% of the baseline's.

| Error | Baseline | DistilBERT |
|---|---|---|
| positive → neutral | 244 | 229 |
| neutral → negative | 131 | 135 |
| negative → neutral | 98 | 98 |
| neutral → positive | 93 | 70 |
| positive → negative | 90 | 34 |
| negative → positive | 45 | 20 |

As on validation, the gain is concentrated in the opposite-polarity mistakes (positive → negative down 62%, negative → positive down 56%), while the errors involving neutral barely move. Neutral → negative is the one type that got slightly worse (131 to 135). Positive → neutral is the largest error for both models, 39% of DistilBERT's test errors.

| Rating | Baseline error | DistilBERT error |
|---|---|---|
| 1 | 27.6% | 20.3% |
| 2 | 46.8% | 39.9% |
| 3 | 53.3% | 48.8% |
| 4 | 28.6% | 28.4% |
| 5 | 6.6% | 2.9% |

Again the middle of the scale is hardest: 3-star reviews are nearly a coin flip even for DistilBERT, and 4-star reviews are misclassified 28% of the time by both models. Reading the samples shows the same two label-noise patterns as on validation:

- **Praise plus a fit complaint, rated 4 or 5, predicted neutral.** "I loved the jumpsuit but unfortunately had to return it due to running a little on the small size" is labelled positive by its 5 stars; the text is mixed.
- **3-star reviews that read as negative.** "...the fabric felt super cheap... not in a good way. returned." and "...it definitely goes back!" are labelled neutral, and the model's "negative" is arguably the better reading of the text.

Length and product group behave as on validation. The error rate rises with review length (9.4% at 20 words or fewer, 20.5% above 100) and varies little across the five departments (14.9% to 19.0%). Class-level differences (for example Jeans 20.5%) rest on small groups and are not interpreted.

**What this means for the report.** The remaining errors are dominated by the neutral class and the 3-to-4-star boundary, where the rating-derived label is itself ambiguous. Further model gains on this label definition are likely to be small; a different label scheme (for example using review text to define "mixed") would be a different task.

## 3. Topic and issue extraction

**Setup.** Both models are fit on the 15,824 training reviews and scored on c_v coherence over a shared tokenized corpus and dictionary. BERTopic embeds raw text with `all-MiniLM-L6-v2` and clusters with UMAP (seeded) and HDBSCAN; its topic words come from a stopword-filtered c-TF-IDF so labels are readable.

| Model | Configuration | Topics | c_v | Outlier reviews |
|---|---|---|---|---|
| LDA | 10 topics | 10 | 0.4701 | n/a |
| BERTopic | min size 50, 10 topics | 9 | 0.4039 | 8.1% |
| **BERTopic (primary)** | **min size 25, auto count** | **53** | **0.4543** | 32.3% |
| BERTopic, near-duplicates merged | same, merged | 45 | 0.4536 | 32.3% |

**Why BERTopic is primary even though LDA scores higher on c_v.** c_v rewards topics whose top words co-occur, and LDA's ten broad topics do that. They are also blurry: their top words are generic ("look, make, fit"). BERTopic at min size 25 surfaces specific, readable topics, including issue topics (itchy wool, sheer white items, missing buttons) that the 10-topic version merges away. The trade-off is 32% of reviews that HDBSCAN leaves unassigned.

**Merging near-duplicates.** The 53-topic fit contained near-duplicates: three jacket topics, three xs/xxs sizing topics, three jeans topics. Topics are merged when every pair in the group has a c-TF-IDF cosine similarity of at least 0.45 (complete linkage), and topics above 500 reviews are never merged, so the large category topics cannot absorb small issue topics. This takes 53 topics to 45 with c_v unchanged (0.4543 → 0.4536) and the same outlier share; the merge makes the topic list shorter and cleaner without costing coherence.

### 3.1 What the topics are

After reading sample reviews for all 45 merged topics, each was labeled by hand (`reports/topic_labels.csv`). Of the 10,716 clustered reviews:

| Kind of topic | Topics | Reviews | Share |
|---|---|---|---|
| Product category (dresses, jeans, jackets, ...) | 24 | 8,247 | 77.0% |
| Sizing and fit | 7 | 1,145 | 10.7% |
| Generic praise | 3 | 776 | 7.2% |
| **Issue** | **5** | **224** | **2.1%** |
| Fabric feel, colour, brand, expectation | 6 | 324 | 3.0% |

Most of what unsupervised topic modelling finds in this corpus is *what the product is*, not *what went wrong*. The five issue topics are: itchy or scratchy wool (70 reviews), layering and sheerness that needs a cami (52), sheerness of white items (44), care instructions and delicacy (30) and buttons that are missing or fall off (28). They are small but specific and actionable, which is the case for BERTopic over LDA. Taken together with the sizing topics, they show that complaints in this data are dominated by fit and sizing rather than product defects.

**Limitation.** The department breakdown is weak for departments whose reviews fall into the large "dress" topic: 84% of Dresses reviews, about 18% of Intimate reviews and 49% of Trend reviews sit in topic 0, so the breakdown mostly tells us what category a department sells. A second-stage topic model inside the largest topics would be the natural next step; it is out of scope here.

## 4. Business interpretation (to expand)

- Fit and sizing are the largest controllable source of dissatisfaction: seven sizing topics hold 11% of clustered reviews, five times the share of the issue topics. The model's most frequent errors are reviews that praise the item but report a fit problem, so a recommendation system built on star ratings alone would misread them.
- The few defect topics are specific enough to route: buttons to quality control, sheerness of white items to product design and a "needs a cami" note on the listing, itchy wool to fabric sourcing.

## 5. Limitations (to expand)

- The test split was used once, as planned, and cannot be reused for further tuning without invalidating it. Validation numbers are slightly optimistic because the model and baseline settings were chosen on that split.
- The Milestone 2 DistilBERT weights were lost, so section 2 error analysis rests on that run's saved validation predictions while the reported model is the final run.
- Sentiment labels come from star ratings; the neutral class and the 4-star boundary are partly label noise.
- A single stratified split; DistilBERT results vary by about 0.006 macro-F1 between runs of the same settings. The only confidence interval computed is the paired bootstrap on the test-set difference, and it reflects test-sample noise, not training-run noise.
- 32% of reviews are BERTopic outliers and are excluded from every topic statistic.
- Topic labels are one reader's judgement and have not been checked by a second reader.
- The dataset is one retailer's women's clothing reviews; results may not carry over to other categories.
