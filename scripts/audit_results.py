"""
Audit script for credit-risk-threshold-lab.
Verifies that all required output files exist, contain valid data,
and that documentation does not contain placeholder text.

Run from the repo root:
    python scripts/audit_results.py
"""

import sys
import os
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)

PASS = "  PASS"
FAIL = "  FAIL"

errors = []
warnings_list = []
checks_run = 0


def check(condition, label, detail=""):
    global checks_run
    checks_run += 1
    if condition:
        print(f"{PASS}  {label}")
    else:
        print(f"{FAIL}  {label}" + (f"  -- {detail}" if detail else ""))
        errors.append(label + (f": {detail}" if detail else ""))


def warn(condition, label, detail=""):
    global checks_run
    checks_run += 1
    if condition:
        print(f"{PASS}  {label}")
    else:
        print(f"  WARN  {label}" + (f"  -- {detail}" if detail else ""))
        warnings_list.append(label)


# -----------------------------------------------------------------------
print("=" * 60)
print("AUDIT: Required figure files")
print("=" * 60)

required_figures = [
    "reports/figures/class_balance.png",
    "reports/figures/logistic_confusion_matrix_t050.png",
    "reports/figures/xgboost_confusion_matrix_t050.png",
    "reports/figures/final_model_confusion_matrix.png",
    "reports/figures/logistic_regression_threshold_metrics.png",
    "reports/figures/xgboost_threshold_metrics.png",
    "reports/figures/expected_cost_by_threshold.png",
    "reports/figures/roc_curve_comparison.png",
    "reports/figures/precision_recall_curve_comparison.png",
    "reports/figures/feature_importance_xgboost.png",
]

for path in required_figures:
    check(os.path.isfile(path), f"Figure exists: {path}",
          "Run scripts/run_analysis.py to generate")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT: Required table CSV files")
print("=" * 60)

required_tables = [
    "reports/tables/class_balance.csv",
    "reports/tables/model_metrics_t050.csv",
    "reports/tables/threshold_sweep_logistic.csv",
    "reports/tables/threshold_sweep_xgboost.csv",
    "reports/tables/final_recommendation.csv",
    "reports/tables/feature_importance_xgboost.csv",
    "reports/tables/logistic_coefficients.csv",
]

for path in required_tables:
    check(os.path.isfile(path), f"Table exists: {path}",
          "Run scripts/run_analysis.py to generate")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT: Processed dataset")
print("=" * 60)

check(os.path.isfile("data/processed/credit_default_clean.csv"),
      "Cleaned dataset exists: data/processed/credit_default_clean.csv")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT: Documentation placeholder check")
print("=" * 60)

PLACEHOLDER_PHRASES = [
    "TODO",
    "placeholder",
    "fill in",
    "TBD",
    "after running",
    "[fill in",
    "[X]",
    "[Date]",
    "[Your Name]",
    "[Stakeholder",
    "fill in after run",
]

for doc_path in ["README.md", "reports/stakeholder_summary.md"]:
    if not os.path.isfile(doc_path):
        check(False, f"Document exists: {doc_path}")
        continue
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    for phrase in PLACEHOLDER_PHRASES:
        found = phrase.lower() in content.lower()
        check(not found, f"No placeholder '{phrase}' in {doc_path}",
              f"Found '{phrase}' -- replace with actual result")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT: final_recommendation.csv integrity")
print("=" * 60)

rec_path = "reports/tables/final_recommendation.csv"
if os.path.isfile(rec_path):
    rec_df = pd.read_csv(rec_path)

    check(len(rec_df) == 1, "final_recommendation.csv has exactly one row",
          f"Found {len(rec_df)} rows")

    if len(rec_df) > 0:
        row = rec_df.iloc[0]

        for col in ["accuracy", "precision", "recall", "f1"]:
            if col in row:
                val = float(row[col])
                check(0.0 <= val <= 1.0, f"final_recommendation {col} in [0, 1]",
                      f"Got {val}")

        for col in ["true_positives", "true_negatives", "false_positives", "false_negatives"]:
            if col in row:
                val = float(row[col])
                check(val >= 0 and val == int(val),
                      f"final_recommendation {col} is nonneg int",
                      f"Got {val}")

        if "threshold" in row:
            t = float(row["threshold"])
            check(0.0 < t < 1.0, f"final_recommendation threshold in (0, 1)",
                  f"Got {t}")

        if "expected_cost" in row:
            ec = float(row["expected_cost"])
            check(ec >= 0, "final_recommendation expected_cost >= 0",
                  f"Got {ec}")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT: model_metrics_t050.csv integrity")
print("=" * 60)

metrics_path = "reports/tables/model_metrics_t050.csv"
if os.path.isfile(metrics_path):
    m_df = pd.read_csv(metrics_path)
    check(len(m_df) >= 2, "model_metrics_t050 has at least 2 model rows",
          f"Found {len(m_df)}")
    for col in ["accuracy", "precision", "recall", "f1"]:
        if col in m_df.columns:
            valid = m_df[m_df["model"] != "Naive (predict 0)"][col].between(0.0, 1.0).all()
            check(valid, f"model_metrics_t050 {col} in [0, 1] for non-naive models")
    for col in ["true_positives", "true_negatives", "false_positives", "false_negatives"]:
        if col in m_df.columns:
            valid = (m_df[col] >= 0).all()
            check(valid, f"model_metrics_t050 {col} all nonneg")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT: threshold sweep integrity")
print("=" * 60)

for name, path in [
    ("logistic", "reports/tables/threshold_sweep_logistic.csv"),
    ("xgboost", "reports/tables/threshold_sweep_xgboost.csv"),
]:
    if os.path.isfile(path):
        sweep_df = pd.read_csv(path)
        check(len(sweep_df) >= 10, f"{name} sweep has >= 10 threshold rows",
              f"Found {len(sweep_df)}")
        if "threshold" in sweep_df.columns:
            check(sweep_df["threshold"].between(0, 1).all(),
                  f"{name} sweep all thresholds in [0, 1]")
        for col in ["false_positives", "false_negatives"]:
            if col in sweep_df.columns:
                check((sweep_df[col] >= 0).all(),
                      f"{name} sweep {col} all nonneg")
        if "expected_cost" in sweep_df.columns:
            check((sweep_df["expected_cost"] >= 0).all(),
                  f"{name} sweep expected_cost all nonneg")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT: class_balance.csv integrity")
print("=" * 60)

cb_path = "reports/tables/class_balance.csv"
if os.path.isfile(cb_path):
    cb_df = pd.read_csv(cb_path)
    check(len(cb_df) == 2, "class_balance has exactly 2 rows", f"Found {len(cb_df)}")
    if "count" in cb_df.columns:
        check((cb_df["count"] > 0).all(), "class_balance counts are positive")
    if "percentage" in cb_df.columns:
        total_pct = cb_df["percentage"].sum()
        check(abs(total_pct - 100.0) < 0.1,
              "class_balance percentages sum to ~100",
              f"Sum = {total_pct:.2f}")


# -----------------------------------------------------------------------
print()
print("=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)

total = checks_run
n_fail = len(errors)
n_warn = len(warnings_list)
n_pass = total - n_fail - n_warn

print(f"Checks run:  {total}")
print(f"Passed:      {n_pass}")
print(f"Failed:      {n_fail}")
print(f"Warnings:    {n_warn}")

if errors:
    print()
    print("Failed checks:")
    for e in errors:
        print(f"  - {e}")

if not errors:
    print()
    print("All checks passed. Repository outputs are complete.")
    sys.exit(0)
else:
    print()
    print("Some checks failed. Run scripts/run_analysis.py and review the failures above.")
    sys.exit(1)
