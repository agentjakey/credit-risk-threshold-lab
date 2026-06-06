# Credit Risk Model Stakeholder Summary

**Prepared for:** [Stakeholder Name / Team]
**Date:** [Date]
**Analyst:** [Your Name]

---

## Decision context

We trained two machine learning models to predict which loan applicants are likely to default. The goal is not to build a perfect predictor but to find the probability cutoff (threshold) that minimizes the total cost of errors, given what we know about what each type of mistake costs the organization.

---

## Positive class

A positive prediction means the model flags the applicant as **high risk of default**. A negative prediction means the model considers the applicant **likely to repay**.

---

## Class imbalance

In this dataset, defaulting borrowers are a minority. Roughly [X]% of the training cases resulted in a default. This matters because a naive model can appear highly accurate simply by predicting everyone will repay. We used techniques (balanced class weights, XGBoost's internal handling) to make the models sensitive to defaults, not just to the majority class.

---

## Why accuracy is not enough

If 90% of borrowers repay, a model that always predicts repayment is 90% accurate on paper. It also catches zero defaults. Accuracy is the wrong scorecard here. We instead evaluate the model on:

- **Recall**: What fraction of actual defaults did we catch?
- **Precision**: Of everyone we flagged as risky, how many were actually risky?
- **Expected cost**: Given our assumptions about what each type of error costs, what threshold minimizes the total bill?

---

## Model comparison

We trained two models and evaluated both across a range of decision thresholds.

| Model | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|
| Logistic Regression | [fill in after run] | [fill in after run] | Interpretable, fast, good baseline |
| XGBoost | [fill in after run] | [fill in after run] | More flexible, often higher recall |

Higher ROC-AUC means the model ranks risky borrowers above safe ones more consistently. PR-AUC is more informative for imbalanced datasets and measures how well the model balances catching defaults against flagging good customers.

---

## Recommended threshold

After sweeping thresholds from 0.10 to 0.90 and applying assumed costs:

- **False positive cost:** $500 per case (a good borrower is denied or offered worse terms)
- **False negative cost:** $5,000 per case (a defaulting borrower receives a loan and does not repay)

**Important:** These cost figures are illustrative assumptions for this exercise. They should be replaced with real estimates from finance, credit risk, or compliance before any operational decision is made.

The selected model and threshold after running the analysis:

- **Model:** [fill in after run]
- **Recommended threshold:** [fill in after run]
- **Precision at this threshold:** [fill in after run]
- **Recall at this threshold:** [fill in after run]
- **Expected cost at this threshold:** [fill in after run]

---

## False positive meaning

When the model predicts default but the borrower would have repaid, this is a false positive. In practice, this means:

- A creditworthy applicant is denied, offered worse terms, or flagged for additional review.
- The organization loses potential revenue and risks customer dissatisfaction.
- At scale, disproportionate false positive rates on any demographic group create regulatory and fairness risk.

---

## False negative meaning

When the model predicts repayment but the borrower defaults, this is a false negative. In practice, this means:

- A loan is extended to a borrower who will not repay.
- The organization absorbs the default loss.
- Because default losses typically far exceed the cost of a denied loan, this is usually the more expensive error.

---

## Business recommendation

Based on the cost assumptions used in this analysis, the recommended operating threshold is [fill in after run]. At this threshold, the model is estimated to reduce expected losses by prioritizing recall on defaults while keeping the false positive rate manageable.

Before using this threshold operationally:

1. Validate the cost assumptions with finance and credit risk teams.
2. Stress test the threshold on held-out data from different time periods.
3. Confirm that false positive rates are equitable across applicant subgroups.
4. Establish a monitoring plan to track model performance after deployment.

---

## Limitations

- The models were trained on [describe your dataset here]. Performance may not generalize to a different applicant population or economic environment.
- The cost assumptions ($500 FP, $5,000 FN) are illustrative. Real costs depend on loan size, interest rate, recovery rate, and operational review costs.
- This model does not incorporate macroeconomic signals, employment history depth, or behavioral data that would be available in a production underwriting system.
- Model performance should be re-evaluated periodically as the borrower population and economic conditions change.

---

## Next steps

1. Replace synthetic cost figures with real estimates from the business.
2. Evaluate the recommended threshold on a fresh holdout set or historical validation period.
3. Conduct a fairness audit on the false positive rate by demographic group.
4. Consider adding SHAP-based feature explanations to support individual loan decisions.
5. Define a monitoring cadence and set alert thresholds for performance degradation.
