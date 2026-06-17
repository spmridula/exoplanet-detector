"""
Two strategies:
  1. RandomizedSearchCV  — fast, good enough for most cases
  2. Manual grid search  — full control, logs every run to MLflow

Why not GridSearchCV?
- With 8 hyperparameters and 5 values each = 5^8 = 390,625 combinations
- RandomizedSearch samples 50-100 random combos — 99% of the benefit
  at 0.01% of the compute cost
"""

import logging
import numpy as np
import pandas as pd
from typing import Any

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Primary scorer: F1 (not accuracy — class imbalance)
F1_SCORER = make_scorer(f1_score, zero_division=0)


def tune_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    RandomizedSearchCV for Random Forest.
    Returns the best params and the fitted best estimator.

    Parameters
    ----------
    n_iter : int
        Number of random parameter combinations to try. Default: 40.
        Increase to 100 for a more thorough search.

    Returns
    -------
    dict with keys: best_params, best_score, cv_results
    """
    from sklearn.ensemble import RandomForestClassifier

    param_dist = {
        "clf__n_estimators":    [100, 200, 300, 500, 750],
        "clf__max_depth":       [5, 8, 12, 15, 20, None],
        "clf__min_samples_leaf":[1, 2, 5, 10, 20],
        "clf__max_features":    ["sqrt", "log2", 0.3, 0.5],
        "clf__class_weight":    ["balanced", "balanced_subsample"],
    }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(random_state=random_state, n_jobs=-1)),
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=F1_SCORER,
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)

    best_params = {k.replace("clf__", ""): v
                   for k, v in search.best_params_.items()}

    logger.info(f"RF best F1: {search.best_score_:.4f}")
    logger.info(f"RF best params: {best_params}")

    return {
        "best_params":   best_params,
        "best_score":    float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results":    pd.DataFrame(search.cv_results_)
                           .sort_values("mean_test_score", ascending=False),
    }


def tune_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    RandomizedSearchCV for XGBoost.
    """
    from xgboost import XGBClassifier

    # Compute scale_pos_weight from data
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw = n_neg / max(n_pos, 1)

    param_dist = {
        "clf__n_estimators":     [100, 200, 300, 500],
        "clf__max_depth":        [3, 4, 5, 6, 8],
        "clf__learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.5, 0.6, 0.8, 1.0],
        "clf__reg_alpha":        [0, 0.01, 0.1, 1.0],
        "clf__reg_lambda":       [0.5, 1.0, 2.0, 5.0],
        "clf__min_child_weight": [1, 3, 5, 10],
    }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=random_state,
            verbosity=0,
        )),
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=F1_SCORER,
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)

    best_params = {k.replace("clf__", ""): v
                   for k, v in search.best_params_.items()}

    logger.info(f"XGB best F1: {search.best_score_:.4f}")
    logger.info(f"XGB best params: {best_params}")

    return {
        "best_params":    best_params,
        "best_score":     float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "cv_results":     pd.DataFrame(search.cv_results_)
                            .sort_values("mean_test_score", ascending=False),
    }


def plot_search_results(
    cv_results: pd.DataFrame,
    param_name: str,
    model_name: str = "Model",
) -> "plt.Figure":
    """
    Scatter plot of a single hyperparameter vs CV F1 score.
    Helps visualize which parameter values work best.
    """
    import matplotlib.pyplot as plt

    col = f"param_clf__{param_name}"
    if col not in cv_results.columns:
        col = f"param_{param_name}"
    if col not in cv_results.columns:
        raise ValueError(f"Column {col} not found in cv_results")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(
        cv_results[col].astype(str),
        cv_results["mean_test_score"],
        alpha=0.6, color="#185FA5", s=40,
    )
    ax.errorbar(
        cv_results[col].astype(str),
        cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
        fmt="none", color="#185FA5", alpha=0.3, capsize=3,
    )
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel("CV F1 Score", fontsize=11)
    ax.set_title(f"{model_name} — {param_name} vs CV F1", fontsize=12, fontweight="normal")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig