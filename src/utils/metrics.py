import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """Pretty-print a metrics dict to stdout."""
    bar = "─" * 40
    print(f"\n{bar}")
    print(f"  {model_name}")
    print(bar)
    print(f"  Precision        : {metrics.get('precision', 0):.4f}")
    print(f"  Recall           : {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score         : {metrics.get('f1', 0):.4f}  ← primary metric")
    print(f"  ROC-AUC          : {metrics.get('roc_auc', 0):.4f}")
    print(f"  PR-AUC           : {metrics.get('average_precision', 0):.4f}")
    print(bar)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Precision-Recall curve — more informative than ROC for imbalanced problems.

    The 'no-skill' baseline is a horizontal line at precision = positive_rate.
    Any model must beat this line to be useful.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    no_skill = y_true.mean()

    ax.plot(recall, precision, lw=2, label=f"{model_name} (AP={ap:.3f})")
    ax.axhline(no_skill, color="gray", linestyle="--", lw=1,
               label=f"No-skill baseline ({no_skill:.3f})")

    ax.set_xlabel("Recall (sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="normal")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if fig:
        plt.tight_layout()
    return fig or ax.get_figure()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    ax: plt.Axes = None,
) -> plt.Figure:
    """ROC curve with AUC annotation."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12, fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if fig:
        plt.tight_layout()
    return fig or ax.get_figure()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> plt.Figure:
    """Confusion matrix with readable labels."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No planet", "Planet"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="normal")
    plt.tight_layout()
    return fig


def plot_model_comparison(results: dict) -> plt.Figure:
    """
    Bar chart comparing multiple models across all metrics.

    Parameters
    ----------
    results : dict of {model_name: metrics_dict}
        e.g. {"Random Forest": {"f1": 0.72, "roc_auc": 0.89, ...}, ...}
    """
    metrics_to_plot = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    labels = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    models = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(models)

    colors = ["#185FA5", "#EF9F27", "#2BAE7E", "#E24B4A", "#9C59D1"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model].get(m, 0) for m in metrics_to_plot]
        bars = ax.bar(x + i * width - (len(models)-1)*width/2,
                      vals, width * 0.9, label=model, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — All Metrics", fontsize=12, fontweight="normal")
    ax.axhline(0.5, color="gray", lw=0.5, linestyle=":")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig