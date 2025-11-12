#!/usr/bin/env python3
"""
Flask web service for local inference.

- Serves index.html
- Exposes:
    GET  /health
    POST /predict
"""

import os
import sys
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# --- Make ../scripts importable ---
APP_DIR = Path(__file__).parent.resolve()
SCRIPTS_DIR = (APP_DIR / "../scripts").resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Import model predictor
import predict as pm

# Flask setup
app = Flask(
    __name__,
    static_folder=str(APP_DIR),
    template_folder=str(APP_DIR),
)


@app.get("/health")
def health():
    return jsonify(pm.get_status())


@app.get("/")
def index():
    return send_from_directory(str(APP_DIR), "index.html")


@app.post("/predict")
def predict_route():
    """
    Expected JSON:
      {
        "model": "xgb" | "logreg",
        "data": { ... }  or  [ {...}, {...} ]
      }
    """
    t0 = time.time()
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        model_key = str(payload.get("model", "xgb")).lower().strip()
        data = payload.get("data", None)
        if data is None:
            return jsonify({"error": "Missing 'data' field in request JSON."}), 400

        result = pm.predict(model_key, data)
        result["latency_ms"] = int((time.time() - t0) * 1000)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
