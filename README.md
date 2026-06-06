# Credit Risk Threshold Lab

A hands-on machine learning project for binary credit-risk classification. The central question is not "how accurate is the model?" but "which threshold should we use, and what does each mistake actually cost?"

This project compares logistic regression and XGBoost on a credit default dataset, then works through threshold selection, class imbalance, confusion matrix interpretation, and a plain-language stakeholder summary.

---

## Why accuracy is not enough

In a dataset where 88% of borrowers do not default, a model that predicts "no default" for everyone is 88% accurate and completely useless. Accuracy rewards the majority class and hides the failures that matter most to a lender.

The real question is: at what probability cutoff do we flag a borrower as high-risk? Lower the threshold and you catch more defaults (higher recall) but flag more good borrowers by mistake (lower precision). Raise it and you stop harassing good customers but miss more actual defaults. This tradeoff has a dollar cost attached to it, and that cost should drive the decision.

---

## Project goal

Build two classifiers on a credit default dataset. Evaluate them not just at the default 0.50 threshold but across a full threshold sweep. Assign realistic (assumed) costs to false positives and false negatives. Use those costs to select the best operating threshold. Communicate the results in a way a non-technical stakeholder can act on.

---

## Dataset instructions

This project does not bundle a proprietary dataset. Place your CSV in `data/raw/` and make sure it has a column named `default` where:

- `0` = borrower did not default (negative class)
- `1` = borrower defaulted (positive class)

If your target column has a different name, rename it to `default` before running.

Good public options:
- UCI Credit Card Default dataset (Taiwan dataset, commonly cited as `default of credit card clients`)
- Kaggle Give Me Some Credit dataset

If no CSV is provided, the notebook includes a synthetic data generator using `sklearn.datasets.make_classification` so you can still run the full pipeline and see every plot.

---

## Key questions answered

1. What does class imbalance look like in this dataset, and why does it matter?
2. How do logistic regression and XGBoost compare on precision, recall, F1, ROC-AUC, and PR-AUC?
3. Why is the precision-recall curve more informative than ROC for imbalanced classification?
4. What happens to precision and recall as we move the threshold from 0.10 to 0.90?
5. Given assumed costs of a false positive ($500) and a false negative ($5,000), which threshold minimizes expected cost?
6. How do we explain these tradeoffs to a stakeholder who does not know what a confusion matrix is?

---

## Models

**Logistic Regression** via a scikit-learn Pipeline with StandardScaler and `class_weight="balanced"`. The balanced weight adjustment compensates for class imbalance by up-weighting the minority class during training.

**XGBoost** with conservative hyperparameters: shallow trees (max_depth=3), low learning rate (0.05), subsampling, and 300 estimators. Trained directly without scaling since gradient boosted trees are scale-invariant.

---

## Metrics

| Metric | Why it matters |
|---|---|
| Accuracy | Baseline check, misleading on imbalanced data |
| Precision | Of predicted defaults, how many are real? |
| Recall | Of actual defaults, how many did we catch? |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Ranking quality across all thresholds |
| PR-AUC | Precision-recall tradeoff, better for imbalanced data |
| Expected cost | Business cost given assumed FP and FN costs |

---

## Business interpretation

A **false negative** means a defaulting borrower was predicted safe and the loan was extended. The lender absorbs the loss.

A **false positive** means a creditworthy borrower was flagged as risky. The loan may be denied or offered at a higher rate. The lender loses a good customer and potentially faces fairness concerns.

These two errors are not equally costly. The project lets you plug in your own cost assumptions and find the threshold that minimizes total expected cost.

---

## How to run

```bash
pip install -r requirements.txt

# Place your dataset at data/raw/your_file.csv
# Then launch the notebook:
jupyter notebook notebooks/01_credit_risk_threshold_lab.ipynb
```

If you have no dataset yet, run the synthetic data cell in the notebook and continue from there.

---

## File structure

```
credit-risk-threshold-lab/
  README.md
  requirements.txt
  data/
    raw/           <- Place your CSV here
    processed/     <- Cleaned data saved here (optional)
  notebooks/
    01_credit_risk_threshold_lab.ipynb
  src/
    config.py      <- Constants and threshold grid
    data.py        <- Load, clean, split, synthetic generator
    modeling.py    <- Logistic regression and XGBoost builders
    evaluation.py  <- Threshold sweep, cost analysis, AUC helpers
    plots.py       <- All figures saved to reports/figures/
  reports/
    stakeholder_summary.md
    figures/       <- Plots saved here
```

---

## Main learning goals

- Understand why accuracy misleads on imbalanced classification problems
- Read and interpret a confusion matrix in business terms
- Understand the precision-recall tradeoff and how threshold changes it
- Use expected cost to make a principled threshold decision
- Communicate model results to a non-technical audience

---

## What I learned

Working through this project made the threshold selection problem concrete. It is easy to say "tune the threshold" but actually sweeping it, visualizing the tradeoffs, and then anchoring the decision to a cost function shows why no single threshold is universally correct. The right threshold depends entirely on what each type of mistake costs your organization.

The comparison between logistic regression and XGBoost also showed that the more powerful model does not always win on the metric that matters most. Both models should be evaluated at the same cost-optimized threshold before declaring a winner.

Working through the stakeholder summary forced a discipline of translating every number into a sentence a non-technical reader can act on.

---

## Internship preparation relevance

Credit risk, fraud detection, loan underwriting, and insurance pricing all share the same structure: binary classification with imbalanced classes and asymmetric error costs. The tools and reasoning in this project apply directly to those domains.

Specifically, this project demonstrates:

- Comfort with the full scikit-learn and XGBoost modeling pipeline
- Understanding of class imbalance and practical mitigations
- Ability to frame model evaluation as a business decision, not just a benchmark
- Stakeholder communication skills grounded in technical results
- Clean, modular Python code organized for collaboration

---

## Customization after running on real data

After running the notebook with a real dataset:

1. Update the cost assumptions ($500 FP, $5,000 FN) to reflect your actual business context.
2. Review the class imbalance section and consider SMOTE or other resampling if the sweep results are unstable.
3. Add feature importance analysis using the SHAP section at the end of the notebook.
4. Fill in the placeholder results in `reports/stakeholder_summary.md` with the actual numbers from your run.
5. Consider adding cross-validation to get more stable threshold estimates.
