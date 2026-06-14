import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.models.baseline import RandomForestModel, XGBoostModel


def make_xy(n=200, n_pos=30, seed=42):
    """Simple separable synthetic feature dataset."""
    np.random.seed(seed)
    n_neg = n - n_pos
    X = pd.DataFrame({
        "dip_depth":   np.concatenate([np.random.uniform(0, 0.002, n_neg),
                                       np.random.uniform(0.008, 0.02, n_pos)]),
        "flux_std":    np.concatenate([np.random.uniform(0.001, 0.003, n_neg),
                                       np.random.uniform(0.004, 0.012, n_pos)]),
        "flux_skew":   np.concatenate([np.random.uniform(-0.2, 0.2, n_neg),
                                       np.random.uniform(-1.5, -0.5, n_pos)]),
        "n_dips":      np.concatenate([np.random.uniform(0, 2, n_neg),
                                       np.random.uniform(5, 15, n_pos)]),
        "period_days": np.concatenate([np.full(n_neg, np.nan),
                                       np.random.uniform(5, 300, n_pos)]),
    })
    y = np.array([0]*n_neg + [1]*n_pos)
    return X, y


class TestRandomForest:
    def test_fit_predict(self):
        X, y = make_xy()
        rf = RandomForestModel(n_estimators=50)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self):
        X, y = make_xy()
        rf = RandomForestModel(n_estimators=50)
        rf.fit(X, y)
        probs = rf.predict_proba(X)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_evaluate_returns_all_metrics(self):
        X, y = make_xy()
        rf = RandomForestModel(n_estimators=50)
        rf.fit(X, y)
        metrics = rf.evaluate(X, y)
        for key in ["precision", "recall", "f1", "roc_auc", "average_precision"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_feature_importances_length(self):
        X, y = make_xy()
        rf = RandomForestModel(n_estimators=50)
        rf.fit(X, y)
        imp = rf.feature_importances()
        assert len(imp) == X.shape[1]
        assert abs(imp.sum() - 1.0) < 1e-6   # importances sum to 1

    def test_f1_beats_naive_baseline(self):
        """Model must beat naive 'predict all negative' baseline on training set."""
        X, y = make_xy(n=300, n_pos=50)
        rf = RandomForestModel(n_estimators=100)
        rf.fit(X, y)
        metrics = rf.evaluate(X, y)
        # Naive baseline F1 for positive class = 0 (never predicts positive)
        assert metrics["f1"] > 0.3


class TestXGBoost:
    def test_fit_and_predict(self):
        X, y = make_xy()
        xgb = XGBoostModel(n_estimators=50)
        xgb.fit(X, y)
        preds = xgb.predict(X)
        assert len(preds) == len(y)

    def test_evaluate_metrics_valid(self):
        X, y = make_xy()
        xgb = XGBoostModel(n_estimators=50)
        xgb.fit(X, y)
        metrics = xgb.evaluate(X, y)
        assert metrics["roc_auc"] > 0.5   # should beat random on training set