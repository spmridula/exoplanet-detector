# src/utils/mlflow_utils.py
"""
mlflow_utils.py
───────────────
MLflow experiment tracking — uses SQLite backend (required in MLflow 3.x,
file store was deprecated). Stores everything in mlflow.db at project root.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exoplanet-detection"
# SQLite backend — works locally, no server needed
TRACKING_URI = "sqlite:///mlflow.db"


def setup_mlflow(tracking_uri: str = None) -> None:
    """Initialize MLflow with SQLite backend. Call once at top of notebook."""
    try:
        import mlflow
        uri = tracking_uri or TRACKING_URI
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow ready — backend: {uri} | experiment: {EXPERIMENT_NAME}")
        print(f"✅ MLflow ready (SQLite backend)")
    except ImportError:
        logger.warning("MLflow not installed — run: pip install mlflow")


@contextmanager
def mlflow_run(run_name: str, params: dict = None):
    """
    Context manager wrapping a training run in MLflow tracking.
    Gracefully skips tracking if MLflow unavailable.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not available — running without tracking")
        yield None
        return

    try:
        with mlflow.start_run(run_name=run_name) as run:
            if params:
                mlflow.log_params(params)
            logger.info(f"MLflow run: {run_name} (id={run.info.run_id[:8]})")
            yield run
    except Exception as e:
        logger.warning(f"MLflow tracking failed ({e}) — continuing without it")
        yield None


def log_metrics(metrics: dict) -> None:
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
    except Exception:
        pass


def log_model(model: Any, artifact_name: str) -> None:
    try:
        import mlflow
        if not mlflow.active_run():
            return
        import pickle, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path=artifact_name)
        os.unlink(tmp_path)
    except Exception as e:
        logger.warning(f"Could not log model: {e}")


def log_figure(fig, filename: str) -> None:
    try:
        import mlflow
        if not mlflow.active_run():
            return
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, dpi=120, bbox_inches="tight")
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="figures")
        os.unlink(tmp_path)
    except Exception as e:
        logger.warning(f"Could not log figure: {e}")