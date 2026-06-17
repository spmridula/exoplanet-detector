# src/utils/feature_importance.py
"""
"Which features actually matter for detecting exoplanets?"

Three methods:
  1. Built-in importance  — fast, from the model directly
  2. Permutation importance — model-agnostic, more reliable
  3. Correlation heatmap  — shows which features are redundant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_builtin_importance(
    model,
    top_n: int = 15,
    title: str = "Feature Importance",
) -> plt.Figure:
    """
    Bar chart of built-in feature importances (RF/XGBoost Gini impurity).

    Fast but biased toward high-cardinality features.
    Use permutation importance for a more honest picture.
    """
    imp = model.feature_importances().head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    colors = ["#EF9F27" if i == 0 else "#185FA5" for i in range(len(imp))]
    imp[::-1].plot(kind="barh", ax=ax, color=colors[::-1], alpha=0.85)

    ax.set_xlabel("Importance (Gini)", fontsize=11)
    ax.set_title(f"{title} — Top {top_n} Features", fontsize=12, fontweight="normal")
    ax.axvline(imp.mean(), color="gray", lw=1, linestyle="--",
               label=f"Mean = {imp.mean():.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    top_n: int = 15,
    n_repeats: int = 10,
    random_state: int = 42,
    title: str = "Permutation Importance",
) -> plt.Figure:
    """
    Permutation importance — shuffles each feature and measures score drop.

    More reliable than built-in importance because:
    - Measures actual impact on predictions, not training impurity
    - Works on the validation set (detects overfitting artifacts)
    - Model-agnostic (same API for RF and XGBoost)

    Slow for large datasets — use n_repeats=5 if too slow.
    """
    from sklearn.metrics import f1_score

    # Use the pipeline's predict method
    def scorer(estimator, X, y):
        return f1_score(y, estimator.predict(X), zero_division=0)

    result = permutation_importance(
        model.pipeline, X_val, y_val,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature":    X_val.columns,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    y_pos = range(len(imp_df))

    ax.barh(list(y_pos), imp_df["importance"][::-1].values,
            xerr=imp_df["std"][::-1].values,
            color="#185FA5", alpha=0.8, capsize=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(imp_df["feature"][::-1].values, fontsize=9)
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Mean F1 decrease when feature shuffled", fontsize=11)
    ax.set_title(f"{title} — Top {top_n} Features", fontsize=12, fontweight="normal")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Features with importance < 0 are actively hurting the model
    n_negative = (imp_df["importance"] < 0).sum()
    if n_negative > 0:
        ax.set_title(
            f"{title}\n({n_negative} features hurt performance — candidates for removal)",
            fontsize=11, fontweight="normal"
        )
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    X: pd.DataFrame,
    figsize: tuple = (11, 9),
) -> plt.Figure:
    """
    Feature correlation heatmap — identifies redundant features.

    Highly correlated features (|r| > 0.8) add noise without information.
    Candidates for removal to reduce model complexity.
    """
    # Only numeric, non-NaN columns
    X_clean = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    corr = X_clean.corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Annotate cells with |r| > 0.6
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            if abs(val) > 0.6 and i != j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.8 else "black")

    ax.set_title("Feature Correlation Matrix\n(|r| > 0.6 annotated)",
                 fontsize=12, fontweight="normal")
    plt.tight_layout()
    return fig


def summarize_importance(
    rf_model,
    xgb_model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Combined importance table: built-in RF rank, built-in XGB rank,
    and permutation rank side by side.

    Helps identify features that are consistently important across methods —
    those are the most trustworthy signal features.
    """
    rf_imp  = rf_model.feature_importances().rename("rf_importance")
    xgb_imp = xgb_model.feature_importances().rename("xgb_importance")

    combined = pd.concat([rf_imp, xgb_imp], axis=1).fillna(0)
    combined["rf_rank"]  = combined["rf_importance"].rank(ascending=False).astype(int)
    combined["xgb_rank"] = combined["xgb_importance"].rank(ascending=False).astype(int)
    combined["avg_rank"] = (combined["rf_rank"] + combined["xgb_rank"]) / 2
    combined = combined.sort_values("avg_rank").head(top_n)

    print(f"\nTop {top_n} features by average rank across RF and XGBoost:")
    print(combined[["rf_rank", "xgb_rank", "avg_rank"]].to_string())
    print("\nFeatures ranked #1 by both models are the strongest signal.")
    return combined