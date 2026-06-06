# Credit Risk Model Stakeholder Summary

**Dataset:** UCI Default of Credit Card Clients (30,000 records; 29,965 after deduplication)  
**Analysis date:** June 2026  
**Models evaluated:** Logistic Regression, XGBoost  
**Final recommendation:** XGBoost at classification threshold 0.20

---

## Decision context

We built two machine learning models to estimate the probability that a credit card customer will default on payment next month. The models were trained on historical payment records from 29,965 customers and evaluated on a held-out test set of 5,993 customers who were not used during training.

The goal is not to build a perfect predictor but to find the operating point -- the probability cutoff -- that minimizes the total cost of errors given what we know about the relative cost of each type of mistake.

---

## Positive class

A positive prediction means the model flags the customer as **high risk of defaulting next month**. A negative prediction means the model considers the customer likely to pay.

---

## Class imbalance

In this dataset, 6,630 of 29,965 customers (22.13%) defaulted in the following month. The remaining 23,335 (77.87%) did not default. This imbalance matters: a model that always predicts "no default" would be correct 77.87% of the time by doing nothing useful. Any evaluation that only looks at overall accuracy will be misled.

Both models were configured with imbalance-aware settings:
- Logistic regression used balanced class weights, which increase the training penalty for misclassifying defaulters.
- XGBoost used a scale_pos_weight of 3.52, computed as the ratio of non-defaulters to defaulters in the training data (18,668 / 5,304 = 3.52).

---

## Why accuracy is not enough

The naive baseline (predict no default for every customer) achieves 77.87% accuracy on the test set. Logistic regression at the default 0.50 threshold achieves only 68.65% accuracy -- lower than doing nothing.

This seems counterintuitive but is not a failure. The logistic regression is configured to catch defaults, which means it predicts default more often. That reduces its overall accuracy while making it actually useful for the task. Accuracy penalizes this behavior unfairly because it treats both types of mistakes as equally bad. They are not.

A missed default (false negative) costs far more than a flagged non-defaulter (false positive). The right scorecard is expected cost under realistic error cost assumptions, not accuracy.

---

## Model comparison at the standard 0.50 threshold

At the most commonly used classification threshold of 0.50:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Naive (predict no default) | 77.87% | N/A | 0% | N/A | N/A |
| Logistic Regression | 68.65% | 37.56% | 62.97% | 47.06% | 0.7162 |
| XGBoost | 75.27% | 45.74% | 63.20% | 53.07% | 0.7757 |

**Recall** is the fraction of actual defaulters the model correctly identified. **Precision** is the fraction of borrowers flagged as risky who actually were risky. XGBoost outperforms logistic regression on every metric.

---

## Why the 0.50 threshold is not optimal

Moving the decision threshold below 0.50 allows the model to flag more borrowers as risky, catching more actual defaults (higher recall) while also flagging more good customers incorrectly (lower precision). Moving it above 0.50 does the opposite.

The right threshold depends on the relative cost of each type of error. We used the following illustrative assumptions for this analysis:

- **False positive cost: $500** -- A creditworthy customer is incorrectly flagged. They may be denied or offered worse terms. The lender loses potential revenue and a satisfied customer.
- **False negative cost: $5,000** -- A defaulting customer is not flagged. The loan is extended and the lender absorbs the loss.

**These cost figures are illustrative assumptions for this exercise only.** They are not based on actual loan sizes, recovery rates, or operational review costs. The right numbers for a production system must come from the credit risk and finance teams.

Expected cost was computed as: (number of false positives x $500) + (number of false negatives x $5,000)

---

## Recommended threshold

After sweeping every threshold from 0.05 to 0.95 for both models, the configuration that minimizes expected cost is:

**XGBoost at classification threshold 0.20**

| Metric | Value |
|---|---|
| Recall (defaults caught) | 96.68% |
| Precision (of flagged customers, truly risky) | 25.15% |
| F1 Score | 0.3992 |
| ROC-AUC | 0.7757 |
| Expected cost on test set | $2,127,500 |

The runner-up is logistic regression at threshold 0.10, with an expected cost of $2,327,000 on the same test set. XGBoost saves approximately $199,500 in expected cost relative to logistic regression under these assumptions.

---

## What the confusion matrix means in plain English

On the 5,993-customer test set, the recommended model (XGBoost at threshold 0.20) produces the following outcomes:

**1,282 true positives** -- Customers who were correctly identified as high-risk. The model flags them; they would have defaulted. Intervention is warranted.

**852 true negatives** -- Customers who were correctly identified as low-risk. The model does not flag them; they would not have defaulted.

**3,815 false positives** -- Customers who were flagged as high-risk but would actually have repaid their loan. These customers are incorrectly treated as risky. At $500 per case, this group represents $1,907,500 in modeled cost.

**44 false negatives** -- Customers who were predicted safe but actually defaulted. These are the loans the model fails to intercept. At $5,000 per case, this group represents $220,000 in modeled loss.

The model catches 1,282 of 1,326 actual defaulters in the test set. It misses 44.

---

## Why the false positive rate is high

The false positive rate at threshold 0.20 is high by design. When the cost of a missed default is ten times the cost of a false alarm, the cost-minimizing decision is to flag aggressively and accept many false positives in exchange for catching nearly all defaults.

If the organization determines that the false positive rate is unacceptably high -- due to customer experience, fairness concerns, or operational review capacity -- the threshold can be raised. The table below shows the tradeoff:

| Threshold | Defaults Caught | Defaults Missed | Non-Defaulters Flagged | Expected Cost |
|---|---|---|---|---|
| 0.20 | 1,282 of 1,326 | 44 | 3,815 of 4,667 | $2,127,500 |
| 0.25 | 1,228 of 1,326 | 98 | 3,281 of 4,667 | $2,130,500 |
| 0.30 | 1,158 of 1,326 | 168 | 2,694 of 4,667 | $2,187,000 |
| 0.35 | 1,076 of 1,326 | 250 | 2,125 of 4,667 | $2,312,500 |
| 0.50 | 838 of 1,326 | 488 | 994 of 4,667 | $2,937,000 |

Moving from threshold 0.20 to 0.25 reduces incorrectly flagged customers from 3,815 to 3,281 (a reduction of 534) while catching 54 fewer defaults. The expected cost difference is only $3,000 under these assumptions. This near-flat cost region between 0.20 and 0.25 means the choice between them should be driven by operational considerations, not cost modeling alone.

---

## Business recommendation

Under the stated cost assumptions, deploy XGBoost at a classification threshold of 0.20. This configuration catches 96.68% of defaults in the test population while accepting a high rate of false alarms.

Before any operational deployment:

1. Replace the illustrative cost figures ($500 FP, $5,000 FN) with estimates grounded in actual loan portfolio data, average default loss, and operational review costs per flagged application.
2. Validate the model on a separate historical cohort from a different time period to confirm that performance does not depend on a specific economic environment.
3. Conduct a fairness audit: confirm that false positive rates are comparable across customer subgroups (by age, marital status, education level) to manage regulatory and reputational risk.
4. Set a monitoring cadence. Model performance should be re-evaluated quarterly and retrained if the default rate or population characteristics shift materially.

---

## Limitations

The model was trained on data from Taiwan in 2005. Performance may not generalize to a different credit market, regulatory environment, or economic cycle.

The 23 features available here (credit limit, demographics, monthly payment status and amounts) are a subset of what a real underwriting system would use. Production models typically incorporate bureau data, employment history, loan purpose, and behavioral signals not present in this dataset.

The cost assumptions are illustrative and do not account for loan size, interest rate, collateral, or recovery rate. Real cost analysis is considerably more complex.

No resampling techniques (SMOTE, undersampling) were used. On more severely imbalanced datasets, resampling can improve minority class recall further.

The model is not calibrated. The predicted probabilities are used only for ranking and threshold selection in this exercise, not as literal probability estimates of default.
