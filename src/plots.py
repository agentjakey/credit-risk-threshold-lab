import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, filename: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_class_balance(y, title="Class Balance"):
    counts = y.value_counts().sort_index()
    labels = ["Non-Default (0)", "Default (1)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["steelblue", "tomato"], width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, "class_balance.png")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred Non-Default", "Pred Default"],
        yticklabels=["Actual Non-Default", "Actual Default"],
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    _save(fig, "confusion_matrix.png")


def plot_precision_recall_by_threshold(sweep_df, model_name="Model"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], marker="o", label="Precision", color="steelblue")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], marker="s", label="Recall", color="tomato")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], marker="^", label="F1", color="seagreen")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 vs Threshold -- {model_name}", fontsize=13)
    ax.legend()
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    filename = f"precision_recall_threshold_{model_name.lower().replace(' ', '_')}.png"
    _save(fig, filename)


def plot_expected_cost_by_threshold(sweep_df, false_positive_cost, false_negative_cost, model_name="Model"):
    df = sweep_df.copy()
    df["expected_cost"] = (
        df["false_positives"] * false_positive_cost
        + df["false_negatives"] * false_negative_cost
    )
    best_idx = df["expected_cost"].idxmin()
    best_threshold = df.loc[best_idx, "threshold"]
    best_cost = df.loc[best_idx, "expected_cost"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["expected_cost"], marker="o", color="darkorange", label="Expected Cost")
    ax.axvline(best_threshold, color="crimson", linestyle="--",
               label=f"Best threshold: {best_threshold:.2f} (cost: {best_cost:,.0f})")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Expected Cost ($)")
    ax.set_title(f"Expected Cost vs Threshold -- {model_name}", fontsize=13)
    ax.legend()
    ax.set_xlim(0.05, 0.95)
    filename = f"expected_cost_threshold_{model_name.lower().replace(' ', '_')}.png"
    _save(fig, filename)


def plot_roc_curve(y_true, y_proba_dict: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    for (label, proba), color in zip(y_proba_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, proba)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend()
    _save(fig, "roc_curve.png")


def plot_precision_recall_curve(y_true, y_proba_dict: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    for (label, proba), color in zip(y_proba_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})", color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend()
    _save(fig, "precision_recall_curve.png")
