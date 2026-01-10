#!/usr/bin/env python3
"""
gRPC client for TensorFlow Serving.

Exposes a `predict` function that:
- takes raw image bytes
- preprocesses to (1, 224, 224, 3) float32
- sends a PredictRequest to TensorFlow Serving
- returns a JSON-serializable dict with predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from preprocessing import preprocess_image
from settings import (
    CLASS_NAMES_PATH,
    INPUT_TENSOR_NAME,
    OUTPUT_TENSOR_NAME,
    TF_SERVING_HOST,
    TF_SERVING_MODEL_NAME,
    TF_SERVING_PORT,
    TF_SERVING_SIGNATURE_NAME,
)


def _load_class_names(path: Path = CLASS_NAMES_PATH) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # Fallback for dict-like mapping
    # {"0": "Bean", "1": "Bitter_Gourd", ...}
    items = sorted(data.items(), key=lambda kv: int(kv[0]))
    return [name for _, name in items]


_CLASS_NAMES: List[str] | None = None


def get_class_names() -> List[str]:
    global _CLASS_NAMES
    if _CLASS_NAMES is None:
        _CLASS_NAMES = _load_class_names()
    return _CLASS_NAMES


def _make_stub() -> prediction_service_pb2_grpc.PredictionServiceStub:
    target = f"{TF_SERVING_HOST}:{TF_SERVING_PORT}"
    channel = grpc.insecure_channel(target)
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)


def predict(image_bytes: bytes, top_k: int = 3) -> Dict[str, Any]:
    """
    Send a prediction request to TensorFlow Serving.

    Returns a JSON-serializable dict:
    {
      "top_prediction": {"class": "...", "probability": 0.95},
      "predictions": [
          {"class": "...", "probability": 0.95},
          ...
      ]
    }
    """
    image_batch = preprocess_image(image_bytes)
    stub = _make_stub()

    request = predict_pb2.PredictRequest()
    request.model_spec.name = TF_SERVING_MODEL_NAME
    request.model_spec.signature_name = TF_SERVING_SIGNATURE_NAME
    request.inputs[INPUT_TENSOR_NAME].CopyFrom(
        tf.make_tensor_proto(image_batch, shape=image_batch.shape)
    )

    response = stub.Predict(request, timeout=5.0)
    outputs = tf.make_ndarray(response.outputs[OUTPUT_TENSOR_NAME])

    probabilities = outputs[0]
    class_names = get_class_names()

    pairs = [
        (class_names[i] if i < len(class_names) else str(i), float(prob))
        for i, prob in enumerate(probabilities)
    ]

    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]

    result: Dict[str, Any] = {
        "top_prediction": {
            "class": top[0][0],
            "probability": top[0][1],
        },
        "predictions": [
            {"class": label, "probability": prob}
            for label, prob in top
        ],
    }

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send image to TF Serving for prediction.")
    parser.add_argument("image_path", help="Path to an image file.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes to return.")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    with img_path.open("rb") as f:
        img_bytes = f.read()

    out = predict(img_bytes, top_k=args.top_k)
    print(json.dumps(out, indent=2))
