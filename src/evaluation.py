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
from src.config import THRESHOLDS


def evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = THRESHOLDS
    rows = [evaluate_at_threshold(y_true, y_proba, t) for t in thresholds]
    return pd.DataFrame(rows)


def expected_cost(row: pd.Series, false_positive_cost: float, false_negative_cost: float) -> float:
    return row["false_positives"] * false_positive_cost + row["false_negatives"] * false_negative_cost


def choose_best_threshold(sweep_df: pd.DataFrame, false_positive_cost: float, false_negative_cost: float) -> dict:
    df = sweep_df.copy()
    df["expected_cost"] = df.apply(
        lambda row: expected_cost(row, false_positive_cost, false_negative_cost),
        axis=1,
    )
    best_row = df.loc[df["expected_cost"].idxmin()]
    return best_row.to_dict()


def compute_roc_auc(y_true, y_proba) -> float:
    return roc_auc_score(y_true, y_proba)


def compute_pr_auc(y_true, y_proba) -> float:
    return average_precision_score(y_true, y_proba)
