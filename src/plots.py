import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


def _save(fig, filename: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_class_balance(y, title="Class Balance -- UCI Credit Default Dataset"):
    counts = y.value_counts().sort_index()
    labels = [f"No Default (0)\nn={counts[0]:,}", f"Default (1)\nn={counts[1]:,}"]
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["steelblue", "tomato"], width=0.45, edgecolor="white")
    for bar, val, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{p:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "class_balance.png")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred: No Default", "Pred: Default"],
        yticklabels=["Actual: No Default", "Actual: Default"],
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    _save(fig, filename)


def plot_threshold_metrics(sweep_df, model_name="Model"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], marker="o", ms=4, label="Precision", color="steelblue")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], marker="s", ms=4, label="Recall", color="tomato")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], marker="^", ms=4, label="F1", color="seagreen")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 vs Threshold -- {model_name}", fontsize=13)
    ax.legend()
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    _save(fig, f"{slug}_threshold_metrics.png")


def plot_expected_cost_by_threshold(lr_sweep, xgb_sweep, fp_cost, fn_cost):
    fig, ax = plt.subplots(figsize=(9, 5))
    for sweep, label, color in [
        (lr_sweep, "Logistic Regression", "steelblue"),
        (xgb_sweep, "XGBoost", "tomato"),
    ]:
        ax.plot(sweep["threshold"], sweep["expected_cost"], marker="o", ms=4, label=label, color=color)
        best_idx = sweep["expected_cost"].idxmin()
        bt = sweep.loc[best_idx, "threshold"]
        bc = sweep.loc[best_idx, "expected_cost"]
        ax.axvline(bt, color=color, linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel(f"Expected Cost ($)  [FP=${fp_cost:,}, FN=${fn_cost:,}]")
    ax.set_title("Expected Cost vs Threshold (Illustrative Assumptions)", fontsize=13)
    ax.legend()
    ax.set_xlim(0.02, 0.98)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    fig.tight_layout()
    _save(fig, "expected_cost_by_threshold.png")


def plot_roc_curve_comparison(y_true, proba_dict: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "tomato"]
    for (label, proba), color in zip(proba_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label}  (AUC = {auc:.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve -- Model Comparison", fontsize=13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "roc_curve_comparison.png")


def plot_pr_curve_comparison(y_true, proba_dict: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "tomato"]
    baseline = y_true.mean()
    for (label, proba), color in zip(proba_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, label=f"{label}  (AP = {ap:.4f})", color=color, linewidth=2)
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve -- Model Comparison", fontsize=13)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "precision_recall_curve_comparison.png")


def plot_feature_importance_xgboost(model, feature_names, top_n=15):
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["importance"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"XGBoost -- Top {top_n} Feature Importances", fontsize=13)
    fig.tight_layout()
    _save(fig, "feature_importance_xgboost.png")
