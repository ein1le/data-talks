#!/usr/bin/env python3
"""
Configuration for the capstone gateway and gRPC client.

Values are centralized here but can still be overridden
via environment variables when running in Docker/Kubernetes.
"""

from __future__ import annotations

import os
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


# Image/model contract
IMG_SIZE = (224, 224)


# TensorFlow Serving configuration
TF_SERVING_HOST = os.getenv("TF_SERVING_HOST", "localhost")
TF_SERVING_PORT = int(os.getenv("TF_SERVING_PORT", "8500"))
TF_SERVING_MODEL_NAME = os.getenv("TF_SERVING_MODEL_NAME", "vegetable-model")
TF_SERVING_SIGNATURE_NAME = os.getenv("TF_SERVING_SIGNATURE_NAME", "serving_default")
INPUT_TENSOR_NAME = os.getenv("TF_SERVING_INPUT_NAME", "image")
OUTPUT_TENSOR_NAME = os.getenv("TF_SERVING_OUTPUT_NAME", "output_0")


# Gateway configuration
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "5000"))

