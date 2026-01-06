#!/usr/bin/env python3
"""
Prediction utilities for Logistic Regression and XGBoost models.

- Loads latest artifacts from ../model_registry
- Transforms input with the saved preprocessing state
- Supports single or batch JSON records
- Exposes: load_models(), predict(), get_status()
"""

import glob
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from helper import (
    preprocess_pipeline_logreg,
    preprocess_pipeline_xgb,
)

# -----------------------
# Paths
# -----------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REGISTRY_DIR = (SCRIPT_DIR / "../model_registry").resolve()

_MODELS: Dict[str, Dict[str, Any]] = {}


# -----------------------
# Internal helpers
# -----------------------
def _latest_artifact(prefix: str) -> Optional[Path]:
    """Return latest artifact path matching prefix."""
    pattern = str(REGISTRY_DIR / f"{prefix}_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: Path(p).name, reverse=True)
    return Path(files_sorted[0])


def _load_artifact(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_dataframe(payload: Any) -> pd.DataFrame:
    """Convert JSON input (dict or list[dict]) to DataFrame."""
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    if isinstance(payload, list):
        if not all(isinstance(x, dict) for x in payload):
            raise ValueError("All list elements must be JSON objects.")
        return pd.DataFrame(payload)
    raise ValueError("Payload must be a JSON object or a list of objects.")


def _transform_for_model(
    df_in: pd.DataFrame,
    model_key: str,
    state: Dict[str, Any],
    expected_cols: List[str],
) -> pd.DataFrame:
    """Apply preprocessing pipeline matching the model."""
    if model_key == "logreg":
        X_out, _ = preprocess_pipeline_logreg(df_in, state)
    elif model_key == "xgb":
        X_out, _ = preprocess_pipeline_xgb(df_in, state)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    return X_out.reindex(columns=expected_cols, fill_value=0.0)


# -----------------------
# Model loading
# -----------------------
def load_models() -> Dict[str, Dict[str, Any]]:
    """Load latest artifacts for both models from the model registry."""
    models: Dict[str, Dict[str, Any]] = {}

    lr_path = _latest_artifact("logreg")
    if lr_path and lr_path.exists():
        art = _load_artifact(lr_path)
        models["logreg"] = {
            "artifact_path": str(lr_path),
            "artifact": art,
            "model": art["model"],
            "state": art["preprocess_state"],
            "features": art.get("feature_columns", []),
            "label_encoder": None,
            "target_name": art.get("target_name", "Segmentation"),
        }

    xgb_path = _latest_artifact("xgb")
    if xgb_path and xgb_path.exists():
        art = _load_artifact(xgb_path)
        models["xgb"] = {
            "artifact_path": str(xgb_path),
            "artifact": art,
            "model": art["model"],
            "state": art["preprocess_state"],
            "features": art.get("feature_columns", []),
            "label_encoder": art.get("label_encoder", None),
            "target_name": art.get("target_name", "Segmentation"),
        }

    return models


_MODELS = load_models()


# -----------------------
# Public API
# -----------------------
def get_status() -> Dict[str, Any]:
    """Expose current model load state."""
    return {
        "status": "ok" if _MODELS else "no models loaded",
        "loaded_models": list(_MODELS.keys()),
        "artifacts": {
            k: v.get("artifact_path", None) for k, v in _MODELS.items()
        },
        "registry_dir": str(REGISTRY_DIR),
    }


def predict(model_key: str, records: Any) -> Dict[str, Any]:
    """Perform predictions using specified model."""
    if not _MODELS:
        raise ValueError("No models loaded. Check model_registry directory.")

    if model_key not in _MODELS:
        raise ValueError(f"Model '{model_key}' not loaded. Available: {list(_MODELS.keys())}")

    entry = _MODELS[model_key]
    model = entry["model"]
    state = entry["state"]
    features = entry["features"]
    label_encoder = entry["label_encoder"]
    artifact_path = entry["artifact_path"]

    df = _to_dataframe(records)
    X = _transform_for_model(df, model_key, state, features)

    y_pred = model.predict(X)

    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if label_encoder is not None:
            classes = label_encoder.inverse_transform(np.arange(proba.shape[1]))
        else:
            classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        y_proba = [
            {str(classes[j]): float(proba[i, j]) for j in range(proba.shape[1])}
            for i in range(proba.shape[0])
        ]

    if label_encoder is not None:
        y_pred_out = label_encoder.inverse_transform(y_pred)
    else:
        y_pred_out = y_pred

    return {
        "model": model_key,
        "version": Path(artifact_path).name,
        "count": int(len(df)),
        "predictions": [str(x) for x in np.atleast_1d(y_pred_out)],
        "probabilities": y_proba,
    }
