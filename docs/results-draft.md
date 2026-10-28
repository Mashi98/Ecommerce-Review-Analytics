# Results draft (Milestone 2 / final report)

Working draft of the results and error-analysis sections. Every number is from a file in `reports/` or from the validation split. The test set has not been used for anything: the Milestone 2 results are validation results (section 1.2 explains why).

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
| DistilBERT | 0.8303 | **0.6823** | 0.8426 |

| Class | Baseline P / R / F1 | DistilBERT P / R / F1 | Support |
|---|---|---|---|
| negative | 0.514 / 0.617 / 0.561 | 0.621 / 0.673 / 0.646 | 358 |
| neutral | 0.359 / 0.467 / 0.406 | 0.405 / 0.563 / 0.472 | 426 |
| positive | 0.945 / 0.872 / 0.907 | 0.967 / 0.895 / 0.929 | 2,633 |

DistilBERT improves every class, and macro-F1 by 0.058. Most of the gain is on the minority classes (negative +0.085 F1, neutral +0.066), which is what macro-F1 rewards. The neutral class stays weak for both models.

### 1.2 Test set: not evaluated at Milestone 2

The 3,386-review test split has not been scored by either model. The reported DistilBERT run (580 validation errors) was trained in a Colab session whose weights were not retained, so it cannot be scored on the test set afterwards. Its validation predictions were exported (`reports/val_preds_distilbert.csv`), which is why every number in this document can still be reproduced from files in the repository.

Retraining to obtain a model that could be scored would not reproduce the reported run (see the note below), so the validation numbers above are the Milestone 2 result. They are slightly optimistic as an estimate of unseen-data performance, because the best DistilBERT checkpoint was chosen on validation macro-F1 and the baseline settings were tuned against the same split. The test split is untouched. The planned single evaluation of both models on it moves to the final report, run with a model whose weights and test predictions are saved at the time.

> Note for the reproducibility section: DistilBERT training on a GPU is not bit-reproducible even with a fixed seed. Two runs with identical settings gave 586 and 580 validation errors (macro-F1 0.676 and 0.682). The reported figures are from the 580-error run; the second run is a measure of run-to-run noise, about 0.006 macro-F1.

## 2. Error analysis

Computed on the validation split only (`reports/error_analysis_baseline.md`, `reports/error_analysis_distilbert.md`).

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

- No test-set result yet (section 1.2). The headline numbers are validation numbers, and the model was selected on that split.
- The reported DistilBERT weights were not retained, so that exact run cannot be re-scored; its validation predictions are saved in `reports/val_preds_distilbert.csv`.
- Sentiment labels come from star ratings; the neutral class and the 4-star boundary are partly label noise.
- A single stratified split; DistilBERT results vary by about 0.006 macro-F1 between runs of the same settings, and no confidence intervals have been computed.
- 32% of reviews are BERTopic outliers and are excluded from every topic statistic.
- Topic labels are one reader's judgement and have not been checked by a second reader.
- The dataset is one retailer's women's clothing reviews; results may not carry over to other categories.
