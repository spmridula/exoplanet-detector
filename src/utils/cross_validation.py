# src/utils/cross_validation.py
"""
cross_validation.py
───────────────────
Week 2, Day 13: Stratified k-fold cross-validation.

Why we need this:
- Our labeled dataset is relatively small (~500 stars in practice)
- A single train/val split gives high-variance estimates
- k-fold gives stable estimates with confidence intervals

We use StratifiedKFold to preserve the class ratio in every fold.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


def cross_validate_model(
    model_class,
    model_params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run stratified k-fold CV and return mean ± std for all metrics.

    Parameters
    ----------
    model_class : RandomForestModel or XGBoostModel class (not instance)
    model_params : dict of constructor kwargs
    X : pd.DataFrame — feature matrix
    y : np.ndarray — binary labels
    n_splits : int — number of folds. Default: 5.

    Returns
    -------
    dict with keys: f1_mean, f1_std, roc_auc_mean, roc_auc_std,
                    ap_mean, ap_std, fold_details
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_f1, fold_auc, fold_ap = [], [], []
    fold_details = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        f1  = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob) if y_val.sum() > 0 else 0.0
        ap  = average_precision_score(y_val, y_prob) if y_val.sum() > 0 else 0.0

        fold_f1.append(f1)
        fold_auc.append(auc)
        fold_ap.append(ap)
        fold_details.append({"fold": fold, "f1": f1, "roc_auc": auc, "ap": ap,
                              "n_pos_val": int(y_val.sum())})

        logger.info(f"  Fold {fold}/{n_splits}: F1={f1:.3f}  AUC={auc:.3f}  AP={ap:.3f}")

    results = {
        "f1_mean":      float(np.mean(fold_f1)),
        "f1_std":       float(np.std(fold_f1)),
        "roc_auc_mean": float(np.mean(fold_auc)),
        "roc_auc_std":  float(np.std(fold_auc)),
        "ap_mean":      float(np.mean(fold_ap)),
        "ap_std":       float(np.std(fold_ap)),
        "fold_details": fold_details,
    }

    logger.info(
        f"CV Result: F1={results['f1_mean']:.3f}±{results['f1_std']:.3f}  "
        f"AUC={results['roc_auc_mean']:.3f}±{results['roc_auc_std']:.3f}"
    )
    return results


def plot_cv_results(cv_results: dict, model_name: str = "Model"):
    """
    Bar chart of per-fold scores with mean line.
    Shows how stable the model is across different data splits.
    """
    import matplotlib.pyplot as plt

    folds = cv_results["fold_details"]
    fold_nums = [f["fold"] for f in folds]
    f1s = [f["f1"] for f in folds]
    aucs = [f["roc_auc"] for f in folds]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, vals, label, color, mean_key, std_key in [
        (axes[0], f1s, "F1 Score", "#185FA5", "f1_mean", "f1_std"),
        (axes[1], aucs, "ROC-AUC",  "#EF9F27", "roc_auc_mean", "roc_auc_std"),
    ]:
        ax.bar(fold_nums, vals, color=color, alpha=0.7, width=0.6)
        mean = cv_results[mean_key]
        std  = cv_results[std_key]
        ax.axhline(mean, color="black", lw=1.5, linestyle="--",
                   label=f"Mean = {mean:.3f} ± {std:.3f}")
        ax.fill_between([0.5, len(folds)+0.5],
                        mean - std, mean + std, alpha=0.15, color=color)
        ax.set_xlabel("Fold", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} per Fold — {model_name}", fontsize=11, fontweight="normal")
        ax.set_xticks(fold_nums)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return fig