import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from src.config import THRESHOLDS, FP_COST, FN_COST


def evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "predicted_positive_rate": round((tp + fp) / n, 6),
    }


def threshold_sweep(y_true, y_proba, thresholds=None, fp_cost=FP_COST, fn_cost=FN_COST) -> pd.DataFrame:
    if thresholds is None:
        thresholds = THRESHOLDS
    rows = []
    for t in thresholds:
        row = evaluate_at_threshold(y_true, y_proba, t)
        row["expected_cost"] = int(row["false_positives"] * fp_cost + row["false_negatives"] * fn_cost)
        rows.append(row)
    return pd.DataFrame(rows)


def choose_best_threshold(sweep_df: pd.DataFrame) -> dict:
    best_idx = sweep_df["expected_cost"].idxmin()
    return sweep_df.loc[best_idx].to_dict()


def compute_roc_auc(y_true, y_proba) -> float:
    return round(roc_auc_score(y_true, y_proba), 6)


def compute_pr_auc(y_true, y_proba) -> float:
    return round(average_precision_score(y_true, y_proba), 6)


def naive_baseline_metrics(y_true) -> dict:
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    return {
        "model": "Naive (always predict 0)",
        "accuracy": round(n_neg / n, 6),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "true_positives": 0,
        "true_negatives": int(n_neg),
        "false_positives": 0,
        "false_negatives": int(n_pos),
    }
