"""
Full analysis pipeline for the Credit Risk Threshold Lab.
Loads the UCI Default of Credit Card Clients dataset, trains both models,
sweeps thresholds, picks the best threshold by expected cost, generates
all figures, and saves all results tables to reports/tables/.

Run from the repo root:
    python scripts/run_analysis.py
"""

import sys
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Ensure repo root is on the path regardless of where this is called from
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.config import (
    TARGET_COL, RANDOM_STATE, THRESHOLDS, FP_COST, FN_COST,
    FIGURES_DIR, TABLES_DIR, FEATURE_NAMES,
)
from src.data import (
    load_uci_credit_dataset, basic_cleaning,
    split_features_target, train_test_split_data, compute_scale_pos_weight,
)
from src.modeling import (
    build_logistic_regression, build_xgboost, train_model, get_predicted_probabilities,
)
from src.evaluation import (
    evaluate_at_threshold, threshold_sweep, choose_best_threshold,
    compute_roc_auc, compute_pr_auc, naive_baseline_metrics,
)
from src.plots import (
    plot_class_balance, plot_confusion_matrix, plot_threshold_metrics,
    plot_expected_cost_by_threshold, plot_roc_curve_comparison,
    plot_pr_curve_comparison, plot_feature_importance_xgboost,
)

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading UCI Default of Credit Card Clients dataset")
print("=" * 60)

df_raw = load_uci_credit_dataset()
print(f"Raw shape: {df_raw.shape}")

df_clean, dupes_dropped, nulls_dropped = basic_cleaning(df_raw)
print(f"After cleaning: {df_clean.shape}  (dropped {dupes_dropped} duplicates, {nulls_dropped} null rows)")

# Save cleaned CSV
clean_path = "data/processed/credit_default_clean.csv"
os.makedirs(os.path.dirname(clean_path), exist_ok=True)
df_clean.to_csv(clean_path, index=False)
print(f"Saved cleaned dataset to {clean_path}")


# -----------------------------------------------------------------------
# 2. Dataset facts
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 2: Dataset facts")
print("=" * 60)

n_rows, n_cols = df_clean.shape
n_features = n_cols - 1
class_counts = df_clean[TARGET_COL].value_counts().sort_index()
class_pcts = (class_counts / len(df_clean) * 100).round(2)
default_rate = df_clean[TARGET_COL].mean()

print(f"Rows: {n_rows:,}")
print(f"Features: {n_features}")
print(f"Target column: {TARGET_COL}")
print(f"Positive class: 1 = default next month")
print(f"Class counts: No Default={class_counts[0]:,}  Default={class_counts[1]:,}")
print(f"Class percentages: No Default={class_pcts[0]:.2f}%  Default={class_pcts[1]:.2f}%")
print(f"Imbalanced: Yes ({class_pcts[1]:.2f}% positive)")

class_balance_df = pd.DataFrame({
    "class": [0, 1],
    "label": ["No Default", "Default"],
    "count": [class_counts[0], class_counts[1]],
    "percentage": [class_pcts[0], class_pcts[1]],
})
class_balance_df.to_csv(f"{TABLES_DIR}/class_balance.csv", index=False)
print(f"Saved: {TABLES_DIR}/class_balance.csv")


# -----------------------------------------------------------------------
# 3. Split
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 3: Train/test split")
print("=" * 60)

X, y = split_features_target(df_clean)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

n_train = len(X_train)
n_test = len(X_test)
train_pos = int(y_train.sum())
train_neg = n_train - train_pos
test_pos = int(y_test.sum())
test_neg = n_test - test_pos

print(f"Train: {n_train:,} rows  ({train_pos:,} defaults, {train_neg:,} non-defaults)")
print(f"Test:  {n_test:,} rows  ({test_pos:,} defaults, {test_neg:,} non-defaults)")
print(f"Train default rate: {y_train.mean():.4f}")
print(f"Test default rate:  {y_test.mean():.4f}")

scale_pos_weight = compute_scale_pos_weight(y_train)
print(f"XGBoost scale_pos_weight (neg/pos in train): {scale_pos_weight}")


# -----------------------------------------------------------------------
# 4. Figures: class balance
# -----------------------------------------------------------------------
plot_class_balance(y)


# -----------------------------------------------------------------------
# 5. Train models
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 4: Training models")
print("=" * 60)

lr = build_logistic_regression()
lr = train_model(lr, X_train, y_train)
print("Logistic Regression trained.")

xgb = build_xgboost(scale_pos_weight=scale_pos_weight)
xgb = train_model(xgb, X_train, y_train)
print("XGBoost trained.")

lr_proba = get_predicted_probabilities(lr, X_test)
xgb_proba = get_predicted_probabilities(xgb, X_test)


# -----------------------------------------------------------------------
# 6. Baseline and metrics at threshold 0.50
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 5: Metrics at threshold 0.50")
print("=" * 60)

naive = naive_baseline_metrics(y_test)
lr_50 = evaluate_at_threshold(y_test, lr_proba, 0.50)
xgb_50 = evaluate_at_threshold(y_test, xgb_proba, 0.50)

lr_roc = compute_roc_auc(y_test, lr_proba)
xgb_roc = compute_roc_auc(y_test, xgb_proba)
lr_pr = compute_pr_auc(y_test, lr_proba)
xgb_pr = compute_pr_auc(y_test, xgb_proba)

rows_50 = []
for name, m, roc, pr in [
    ("Naive (predict 0)", naive, 0.0, 0.0),
    ("Logistic Regression", lr_50, lr_roc, lr_pr),
    ("XGBoost", xgb_50, xgb_roc, xgb_pr),
]:
    rows_50.append({
        "model": name,
        "threshold": m.get("threshold", 0.50),
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "roc_auc": roc,
        "pr_auc": pr,
        "true_positives": m["true_positives"],
        "true_negatives": m["true_negatives"],
        "false_positives": m["false_positives"],
        "false_negatives": m["false_negatives"],
    })

metrics_df = pd.DataFrame(rows_50)
metrics_df.to_csv(f"{TABLES_DIR}/model_metrics_t050.csv", index=False)
print(metrics_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

# Confusion matrix plots at t=0.50
lr_pred_50 = (lr_proba >= 0.50).astype(int)
xgb_pred_50 = (xgb_proba >= 0.50).astype(int)
plot_confusion_matrix(y_test, lr_pred_50,
                      title="Logistic Regression -- Threshold 0.50",
                      filename="logistic_confusion_matrix_t050.png")
plot_confusion_matrix(y_test, xgb_pred_50,
                      title="XGBoost -- Threshold 0.50",
                      filename="xgboost_confusion_matrix_t050.png")


# -----------------------------------------------------------------------
# 7. ROC and PR curves
# -----------------------------------------------------------------------
proba_dict = {
    "Logistic Regression": lr_proba,
    "XGBoost": xgb_proba,
}
plot_roc_curve_comparison(y_test, proba_dict)
plot_pr_curve_comparison(y_test, proba_dict)


# -----------------------------------------------------------------------
# 8. Threshold sweep
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 6: Threshold sweep (0.05 to 0.95)")
print("=" * 60)

lr_sweep = threshold_sweep(y_test, lr_proba, THRESHOLDS, fp_cost=FP_COST, fn_cost=FN_COST)
xgb_sweep = threshold_sweep(y_test, xgb_proba, THRESHOLDS, fp_cost=FP_COST, fn_cost=FN_COST)

lr_sweep.to_csv(f"{TABLES_DIR}/threshold_sweep_logistic.csv", index=False)
xgb_sweep.to_csv(f"{TABLES_DIR}/threshold_sweep_xgboost.csv", index=False)
print(f"Saved sweep tables.")

plot_threshold_metrics(lr_sweep, model_name="Logistic Regression")
plot_threshold_metrics(xgb_sweep, model_name="XGBoost")
plot_expected_cost_by_threshold(lr_sweep, xgb_sweep, FP_COST, FN_COST)


# -----------------------------------------------------------------------
# 9. Best threshold selection
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 7: Best threshold by expected cost")
print("=" * 60)

lr_best = choose_best_threshold(lr_sweep)
xgb_best = choose_best_threshold(xgb_sweep)

print(f"LR  best threshold: {lr_best['threshold']}  expected cost: ${lr_best['expected_cost']:,.0f}")
print(f"XGB best threshold: {xgb_best['threshold']}  expected cost: ${xgb_best['expected_cost']:,.0f}")

lr_best["model"] = "Logistic Regression"
lr_best["roc_auc"] = lr_roc
lr_best["pr_auc"] = lr_pr

xgb_best["model"] = "XGBoost"
xgb_best["roc_auc"] = xgb_roc
xgb_best["pr_auc"] = xgb_pr

# Pick winner: lowest expected cost
if xgb_best["expected_cost"] <= lr_best["expected_cost"]:
    winner = xgb_best
    winner_proba = xgb_proba
    winner_name = "XGBoost"
else:
    winner = lr_best
    winner_proba = lr_proba
    winner_name = "Logistic Regression"

print()
print(f"RECOMMENDED MODEL: {winner_name}")
print(f"RECOMMENDED THRESHOLD: {winner['threshold']}")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
          "true_positives", "true_negatives", "false_positives", "false_negatives", "expected_cost"]:
    print(f"  {k}: {winner[k]}")

# Confusion matrix for final recommendation
final_pred = (winner_proba >= winner["threshold"]).astype(int)
plot_confusion_matrix(
    y_test, final_pred,
    title=f"{winner_name} -- Threshold {winner['threshold']} (Recommended)",
    filename="final_model_confusion_matrix.png",
)

# Save final recommendation
rec_df = pd.DataFrame([winner])
rec_df["selected"] = True
rec_df.to_csv(f"{TABLES_DIR}/final_recommendation.csv", index=False)
print(f"Saved: {TABLES_DIR}/final_recommendation.csv")


# -----------------------------------------------------------------------
# 10. Feature importance
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 8: Feature importance and interpretability")
print("=" * 60)

feature_names = list(X_train.columns)

# XGBoost feature importance
importances = xgb.feature_importances_
fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)
fi_df.to_csv(f"{TABLES_DIR}/feature_importance_xgboost.csv", index=False)
print("Top 10 XGBoost features:")
print(fi_df.head(10).to_string(index=False))
plot_feature_importance_xgboost(xgb, feature_names, top_n=15)

# Logistic regression coefficients
lr_clf = lr.named_steps["clf"]
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_clf.coef_[0],
}).sort_values("coefficient", ascending=False)
coef_df.to_csv(f"{TABLES_DIR}/logistic_coefficients.csv", index=False)
print("\nTop 5 positive LR coefficients (associated with higher predicted default risk):")
print(coef_df.head(5).to_string(index=False))
print("\nTop 5 negative LR coefficients (associated with lower predicted default risk):")
print(coef_df.tail(5).to_string(index=False))


# -----------------------------------------------------------------------
# 11. SHAP (optional, graceful fallback)
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 9: SHAP (optional)")
print("=" * 60)

try:
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved: {shap_path}")
except Exception as e:
    print(f"SHAP skipped: {e}")
    print("Feature importance plot is used instead.")


# -----------------------------------------------------------------------
# 12. Summary printout for copy-paste into docs
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("ANALYSIS COMPLETE -- KEY NUMBERS FOR DOCUMENTATION")
print("=" * 60)
print(f"Dataset rows (after cleaning): {len(df_clean):,}")
print(f"Features: {n_features}")
print(f"No-default count: {class_counts[0]:,} ({class_pcts[0]:.2f}%)")
print(f"Default count:    {class_counts[1]:,} ({class_pcts[1]:.2f}%)")
print(f"Train rows: {n_train:,}  Test rows: {n_test:,}")
print(f"scale_pos_weight (XGBoost): {scale_pos_weight}")
print()
print(f"Naive baseline accuracy:        {naive['accuracy']:.4f}")
print(f"LR  accuracy at t=0.50:         {lr_50['accuracy']:.4f}")
print(f"XGB accuracy at t=0.50:         {xgb_50['accuracy']:.4f}")
print(f"LR  ROC-AUC:                    {lr_roc:.4f}")
print(f"XGB ROC-AUC:                    {xgb_roc:.4f}")
print(f"LR  PR-AUC:                     {lr_pr:.4f}")
print(f"XGB PR-AUC:                     {xgb_pr:.4f}")
print()
print(f"LR  best threshold:             {lr_best['threshold']}")
print(f"LR  precision at best t:        {lr_best['precision']:.4f}")
print(f"LR  recall at best t:           {lr_best['recall']:.4f}")
print(f"LR  f1 at best t:               {lr_best['f1']:.4f}")
print(f"LR  expected cost at best t:    ${lr_best['expected_cost']:,.0f}")
print()
print(f"XGB best threshold:             {xgb_best['threshold']}")
print(f"XGB precision at best t:        {xgb_best['precision']:.4f}")
print(f"XGB recall at best t:           {xgb_best['recall']:.4f}")
print(f"XGB f1 at best t:               {xgb_best['f1']:.4f}")
print(f"XGB expected cost at best t:    ${xgb_best['expected_cost']:,.0f}")
print()
print(f"FINAL RECOMMENDATION: {winner_name} at threshold {winner['threshold']}")
print(f"  Accuracy:          {winner['accuracy']:.4f}")
print(f"  Precision:         {winner['precision']:.4f}")
print(f"  Recall:            {winner['recall']:.4f}")
print(f"  F1:                {winner['f1']:.4f}")
print(f"  ROC-AUC:           {winner['roc_auc']:.4f}")
print(f"  PR-AUC:            {winner['pr_auc']:.4f}")
print(f"  True positives:    {winner['true_positives']}")
print(f"  True negatives:    {winner['true_negatives']}")
print(f"  False positives:   {winner['false_positives']}")
print(f"  False negatives:   {winner['false_negatives']}")
print(f"  Expected cost:     ${winner['expected_cost']:,.0f}")
print(f"  (FP cost=${FP_COST:,}, FN cost=${FN_COST:,} -- illustrative assumptions)")
print()
print("All figures saved to:", FIGURES_DIR)
print("All tables saved to:", TABLES_DIR)
