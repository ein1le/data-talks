#!/usr/bin/env python3
"""
Minimal Flask service for local inference.

- Loads the latest Logistic Regression and XGBoost artifacts from ../model_registry
- Applies the exact stored preprocessing state before prediction
- Supports single-record or batch JSON
- Serves index.html for a tiny local UI
- Exposes /health and /predict routes
"""

import os
import glob
import json
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# Import your notebook-derived helper functions
from helper import (
    preprocess_pipeline_logreg,
    preprocess_pipeline_xgb,
)

# -----------------------
# App + config
# -----------------------
APP_DIR = Path(__file__).parent.resolve()
REGISTRY_DIR = (APP_DIR / "../model_registry").resolve()

app = Flask(
    __name__,
    static_folder=str(APP_DIR),
    template_folder=str(APP_DIR),
)

# -----------------------
# Model registry helpers
# -----------------------
def _latest_artifact(prefix: str) -> Optional[Path]:
    """
    Find the latest artifact by filename timestamp, e.g. 'logreg_20251112_153334.pkl'
    """
    pattern = str(REGISTRY_DIR / f"{prefix}_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: Path(p).name, reverse=True)
    return Path(files_sorted[0])

def _load_artifact(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_all_models() -> Dict[str, Dict[str, Any]]:
    """
    Load latest LR and XGB artifacts (if available).
    Returns dict like:
      {
        "logreg": {"artifact": ..., "model": ..., "state": ..., "features": [...]},
        "xgb":    {"artifact": ..., "model": ..., "state": ..., "features": [...], "label_encoder": ...}
      }
    """
    models: Dict[str, Dict[str, Any]] = {}

    # Logistic Regression
    lr_path = _latest_artifact("logreg")
    if lr_path and lr_path.exists():
        art = _load_artifact(lr_path)
        models["logreg"] = {
            "artifact_path": str(lr_path),
            "artifact": art,
            "model": art["model"],
            "state": art["preprocess_state"],
            "features": art.get("feature_columns", []),
            "label_encoder": None,  # not used
            "target_name": art.get("target_name", "Segmentation"),
        }

    # XGBoost
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

MODELS = _load_all_models()

# -----------------------
# Preprocess + predict helpers
# -----------------------
def _to_dataframe(payload: Any) -> pd.DataFrame:
    """
    Accept a dict (single record) or list[dict] (batch) and coerce to DataFrame.
    """
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    if isinstance(payload, list):
        if not all(isinstance(x, dict) for x in payload):
            raise ValueError("For list payloads, each item must be an object/dict.")
        return pd.DataFrame(payload)
    raise ValueError("Payload must be a JSON object or a list of JSON objects.")

def _transform_for_model(
    df_in: pd.DataFrame,
    model_key: str,
    state: Dict[str, Any],
    expected_cols: List[str],
) -> pd.DataFrame:
    """
    Use the correct transform path for the chosen model, reindex to the training
    feature order, and fill missing with 0 (matching training-time behavior).
    """
    if model_key == "logreg":
        X_out, _ = preprocess_pipeline_logreg(df_in, state)
    elif model_key == "xgb":
        X_out, _ = preprocess_pipeline_xgb(df_in, state)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # Reindex to training feature order (training code guarantees this)
    if expected_cols:
        X_out = X_out.reindex(columns=expected_cols, fill_value=0.0)

    return X_out

def _predict(
    model_key: str,
    records: Any,
) -> Dict[str, Any]:
    """
    Main prediction routine:
      - validate model exists
      - coerce input JSON to DataFrame
      - transform using saved preprocessing state
      - predict and (optionally) probabilities
      - map labels back for XGB via LabelEncoder if available
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Model '{model_key}' is not loaded. "
            f"Available: {', '.join(MODELS.keys()) or '(none)'}"
        )

    entry = MODELS[model_key]
    model = entry["model"]
    state = entry["state"]
    features = entry["features"]
    label_encoder = entry["label_encoder"]
    artifact_path = entry["artifact_path"]

    df = _to_dataframe(records)
    X = _transform_for_model(df, model_key, state, features)

    # Predict
    y_pred = model.predict(X)
    y_proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # If label encoder exists (XGB), map class indices to original labels for clarity
            if label_encoder is not None:
                class_labels = list(label_encoder.inverse_transform(np.arange(proba.shape[1])))
            else:
                # Derive classes from model if possible
                class_labels = list(getattr(model, "classes_", np.arange(proba.shape[1])))
            y_proba = [
                {str(class_labels[j]): float(proba[i, j]) for j in range(proba.shape[1])}
                for i in range(proba.shape[0])
            ]
    except Exception:
        # It's fine if model doesn't support proba
        y_proba = None

    # Inverse-transform predictions for XGB if needed
    if label_encoder is not None:
        y_pred_out = label_encoder.inverse_transform(y_pred)
    else:
        y_pred_out = y_pred

    return {
        "model": model_key,
        "version": os.path.basename(artifact_path),
        "count": int(len(df)),
        "predictions": [str(x) for x in np.atleast_1d(y_pred_out)],
        "probabilities": y_proba,  # may be None
    }

# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "loaded_models": list(MODELS.keys()),
        "registry_dir": str(REGISTRY_DIR),
    })

@app.get("/")
def index():
    # Serve your local index.html (kept in the same folder as app.py)
    return send_from_directory(str(APP_DIR), "index.html")

@app.post("/predict")
def predict():
    """
    Expect JSON of the form:
      {
        "model": "xgb" | "logreg",
        "data": { ... }           # single record
        -- or --
        "data": [ {..}, {..} ]    # batch
      }

    Returns predictions (+ probabilities if available).
    """
    t0 = time.time()
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        model_key = str(payload.get("model", "xgb")).lower().strip()
        data = payload.get("data", None)
        if data is None:
            return jsonify({"error": "Missing 'data' in request JSON."}), 400

        result = _predict(model_key, data)
        result["latency_ms"] = int((time.time() - t0) * 1000)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Log server-side if you want; return generic message to client
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.post("/reload")
def reload_models():
    """
    Optional helper to hot-reload the newest artifacts from the registry.
    """
    global MODELS
    MODELS = _load_all_models()
    return jsonify({
        "status": "reloaded",
        "loaded_models": list(MODELS.keys())
    })

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    # For local dev only; in production use gunicorn:
    #   gunicorn -w 2 -b 0.0.0.0:8000 app:app
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
