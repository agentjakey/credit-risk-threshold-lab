# Credit Risk Threshold Lab

A hands-on machine learning project for binary credit-risk classification using a real public dataset. The central question is not "how accurate is the model?" but "which threshold minimizes the cost of mistakes?"

This project trains logistic regression and XGBoost on the UCI Default of Credit Card Clients dataset, sweeps decision thresholds from 0.05 to 0.95, attaches business costs to false positives and false negatives, and selects the operating point that minimizes total expected cost. All results come from executed code on the real dataset.

---

## Why accuracy is not enough

The dataset has 22.13% defaulters and 77.87% non-defaulters. A model that predicts "no default" for every applicant achieves 77.87% accuracy and catches zero actual defaults. That is the naive baseline -- and it is the bar any real model must beat meaningfully, not just numerically.

Accuracy rewards the majority class. In credit risk, the minority class (defaults) is exactly what we care about catching. The right question is not "what fraction of predictions are correct overall?" but "given what each type of mistake costs, at what threshold should we flag a borrower as high-risk?"

---

## Dataset

**Source:** UCI Default of Credit Card Clients (ID 350)  
**Citation:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications.

| Fact | Value |
|---|---|
| Raw rows | 30,000 |
| Rows after deduplication | 29,965 |
| Duplicates dropped | 35 |
| Missing values | 0 |
| Features | 23 |
| Target column | `default` (renamed from `Y`) |
| Positive class | 1 = default next month |
| Negative class | 0 = no default next month |

**Class balance:**

| Class | Count | Percentage |
|---|---|---|
| No Default (0) | 23,335 | 77.87% |
| Default (1) | 6,630 | 22.13% |

**Train / test split (80/20, stratified):**

| Split | Rows | Defaults | Non-defaults |
|---|---|---|---|
| Train | 23,972 | 5,304 | 18,668 |
| Test | 5,993 | 1,326 | 4,667 |

**Features include:** credit limit (LIMIT_BAL), demographics (SEX, EDUCATION, MARRIAGE, AGE), repayment status for 6 months (PAY_0 through PAY_6), bill amounts for 6 months (BILL_AMT1-6), and payment amounts for 6 months (PAY_AMT1-6).

**Data instructions:** To reproduce this analysis, run `scripts/run_analysis.py`. The script fetches the dataset automatically via `ucimlrepo` and saves the cleaned version to `data/processed/credit_default_clean.csv`.

---

## Models

**Logistic Regression:** scikit-learn Pipeline with StandardScaler and `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`. Balanced class weights adjust the training loss so the minority class receives proportionally more weight during fitting, compensating for imbalance without resampling.

**XGBoost:** `XGBClassifier` with `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`, and `scale_pos_weight=3.5196`. The `scale_pos_weight` is computed as the ratio of negative to positive training samples (18,668 / 5,304 = 3.5196), which performs a similar function to balanced class weights for gradient boosted trees.

---

## Baseline comparison

At the default 0.50 threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Naive (always predict 0) | 0.7787 | 0.0000 | 0.0000 | 0.0000 | -- | -- |
| Logistic Regression | 0.6865 | 0.3756 | 0.6297 | 0.4706 | 0.7162 | 0.5015 |
| XGBoost | 0.7527 | 0.4574 | 0.6320 | 0.5307 | 0.7757 | 0.5557 |

The logistic regression accuracy at 0.50 (68.65%) falls below the naive baseline (77.87%). This is expected: balanced class weights train the model to predict more defaults, which reduces accuracy by the standard definition while making the model actually useful for catching them. Accuracy alone would tell you logistic regression is worse than doing nothing. That conclusion is wrong.

XGBoost outperforms logistic regression on every metric at t=0.50, with a ROC-AUC of 0.7757 versus 0.7162 and a PR-AUC of 0.5557 versus 0.5015. PR-AUC is the more informative metric here because it focuses on the minority class performance across all thresholds.

---

## Threshold selection

We swept thresholds from 0.05 to 0.95 in steps of 0.05 and computed expected cost at each threshold using illustrative assumed costs:

- **False positive cost: $500** (a creditworthy borrower is flagged as risky -- potential lost business, operational review cost)
- **False negative cost: $5,000** (a defaulting borrower receives a loan they do not repay -- lender absorbs the loss)

These cost figures are example assumptions for this exercise. They are not real lender values. Actual cost analysis requires input from credit risk, finance, and compliance teams.

**Expected cost = (false positives x $500) + (false negatives x $5,000)**

| Model | Best Threshold | Precision | Recall | F1 | False Positives | False Negatives | Expected Cost |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.10 | 0.2234 | 0.9947 | 0.3649 | 4,584 | 7 | $2,327,000 |
| XGBoost | 0.20 | 0.2515 | 0.9668 | 0.3992 | 3,815 | 44 | $2,127,500 |

XGBoost at threshold 0.20 produces the lower expected cost ($2,127,500 versus $2,327,000 for logistic regression) and is the recommended configuration.

---

## Final recommendation

**Model:** XGBoost  
**Threshold:** 0.20  
**Cost assumptions:** FP=$500, FN=$5,000 (illustrative only)

| Metric | Value |
|---|---|
| Accuracy | 0.3561 |
| Precision | 0.2515 |
| Recall | 0.9668 |
| F1 | 0.3992 |
| ROC-AUC | 0.7757 |
| PR-AUC | 0.5557 |
| True positives | 1,282 |
| True negatives | 852 |
| False positives | 3,815 |
| False negatives | 44 |
| Expected cost (test set) | $2,127,500 |

On the test set of 5,993 borrowers, this configuration catches 1,282 of 1,326 actual defaults (96.68% recall) while missing only 44. It incorrectly flags 3,815 of 4,667 non-defaulting borrowers, resulting in a low precision of 25.15%. The 10:1 cost ratio ($5,000 FN vs $500 FP) drives the optimizer to heavily favor recall, accepting a high false positive rate to nearly eliminate missed defaults.

This tradeoff is correct under these cost assumptions. If false positive costs are higher in practice (for example, due to regulatory concern about denying credit to qualified borrowers), raising the threshold to 0.25 or 0.30 would reduce false positives substantially at a modest increase in expected cost.

| Threshold (XGB) | Precision | Recall | False Positives | False Negatives | Expected Cost |
|---|---|---|---|---|---|
| 0.20 | 0.2515 | 0.9668 | 3,815 | 44 | $2,127,500 |
| 0.25 | 0.2723 | 0.9261 | 3,281 | 98 | $2,130,500 |
| 0.30 | 0.3006 | 0.8733 | 2,694 | 168 | $2,187,000 |
| 0.35 | 0.3361 | 0.8115 | 2,125 | 250 | $2,312,500 |
| 0.50 | 0.4574 | 0.6320 | 994 | 488 | $2,937,000 |

---

## Feature importance

**XGBoost top features (by gain):**

| Feature | Importance |
|---|---|
| PAY_0 (Sept repayment status) | 0.3530 |
| PAY_2 (Aug repayment status) | 0.1175 |
| PAY_3 (July repayment status) | 0.0751 |
| PAY_4 (June repayment status) | 0.0590 |
| PAY_6 (April repayment status) | 0.0432 |

Recent payment behavior (PAY_0, PAY_2) dominates model predictions by a wide margin, which aligns with domain intuition: a borrower who has already missed multiple payments is much more likely to default next month.

**Logistic regression coefficients (top associated with higher predicted default risk):**

| Feature | Coefficient |
|---|---|
| PAY_0 | +0.5787 |
| BILL_AMT2 | +0.2579 |
| PAY_2 | +0.1196 |

**Top associated with lower predicted default risk:**

| Feature | Coefficient |
|---|---|
| BILL_AMT1 | -0.4163 |
| PAY_AMT1 | -0.2419 |
| PAY_AMT2 | -0.1662 |

These are associations in the model, not causal claims. BILL_AMT1 and PAY_AMT1 appearing on opposite ends is consistent with a pattern where having a large bill but making a payment is a different signal than having a large bill without paying.

---

## Business interpretation

A **false negative** means the model predicted no default, the loan was approved, and the borrower defaulted. The lender absorbs the unpaid principal. Under the example cost assumptions ($5,000 each), 44 false negatives in the test set represent $220,000 in modeled losses.

A **false positive** means the model predicted default, but the borrower would have repaid. The loan may be denied or offered at worse terms. The lender loses a good customer and risks fairness and regulatory scrutiny. Under the example assumptions ($500 each), 3,815 false positives in the test set represent $1,907,500 in modeled opportunity cost and operational cost.

The total expected cost of $2,127,500 on the test set reflects these illustrative assumptions and should not be interpreted as a real financial projection.

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis (downloads dataset, trains models, generates all figures and tables)
python scripts/run_analysis.py

# Run audit to verify all outputs were generated correctly
python scripts/audit_results.py

# Launch the notebook for the full guided walkthrough
jupyter notebook notebooks/01_credit_risk_threshold_lab.ipynb
```

---

## Project structure

```
credit-risk-threshold-lab/
  README.md
  requirements.txt
  scripts/
    run_analysis.py        <- Full pipeline runner
    audit_results.py       <- Output integrity checker
  data/
    processed/             <- Cleaned CSV saved here by run_analysis.py
  notebooks/
    01_credit_risk_threshold_lab.ipynb
  src/
    config.py              <- Constants, threshold grid, cost assumptions
    data.py                <- Load, clean, split functions
    modeling.py            <- LR and XGBoost builders
    evaluation.py          <- Threshold sweep, cost analysis, AUC helpers
    plots.py               <- Figure generators
  reports/
    stakeholder_summary.md <- Business-facing write-up
    figures/               <- All PNG outputs
    tables/                <- All CSV result tables
```

---

## Main learning goals

- Understand why accuracy misleads on imbalanced binary classification problems
- Read and interpret a confusion matrix in business terms
- Understand the precision-recall tradeoff and how threshold changes it
- Use expected cost to make a principled threshold decision
- Compare ROC-AUC and PR-AUC and understand when each is appropriate
- Communicate model results to a non-technical stakeholder

---

## What I learned

The naive baseline is the sharpest illustration of the accuracy problem. 77.87% accuracy sounds reasonable. It is achieved by a model that has learned nothing, predicts nothing useful, and would approve every loan regardless of risk. Any model that beats this number on accuracy alone is not necessarily better -- the logistic regression at t=0.50 scores below the naive baseline on accuracy (68.65%) while being dramatically more useful for the actual task.

The threshold sweep showed that the cost-minimizing threshold is not 0.50 for either model. The optimal XGBoost threshold of 0.20 catches 96.68% of defaults at the cost of a very high false positive rate. Whether that tradeoff is right depends entirely on the actual ratio of default loss to review cost -- which is a business question, not a modeling question.

The most important feature by a large margin is PAY_0, the repayment status in the most recent month. This makes intuitive sense and is reassuring: the model is not relying on demographic proxies but on payment behavior.

---

## Internship preparation relevance

Credit risk, fraud detection, loan underwriting, and insurance pricing share the same structure: binary classification with imbalanced classes and asymmetric error costs. The tools and reasoning in this project apply directly to those domains.

This project demonstrates comfort with the full scikit-learn and XGBoost pipeline, understanding of class imbalance and practical mitigations (balanced weights, scale_pos_weight), ability to frame model evaluation as a business decision rather than a benchmark exercise, and the discipline to translate every number into a sentence a stakeholder can act on.
