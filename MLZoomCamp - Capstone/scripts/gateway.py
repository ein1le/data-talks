#!/usr/bin/env python3
"""
Flask gateway that accepts image uploads and forwards them
to a TensorFlow Serving backend via the gRPC client in predict.py.

Endpoint:
  POST /predict
    form-data: file=<image>
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import predict
from settings import GATEWAY_PORT


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__)


@app.post("/predict")
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data."}), 400

    file_storage = request.files["file"]
    if not file_storage.filename:
        return jsonify({"error": "Uploaded file has no filename."}), 400

    image_bytes = file_storage.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = predict.predict(image_bytes)
        return jsonify(result), 200
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "Prediction failed.", "detail": str(exc)}), 500


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=GATEWAY_PORT, debug=False)
