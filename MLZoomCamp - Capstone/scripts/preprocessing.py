#!/usr/bin/env python3
"""
Image preprocessing utilities for the capstone project.

Responsibilities:
- load image bytes
- convert to RGB
- resize to the configured IMG_SIZE
- normalize pixel values to [0, 1]
- add a batch dimension for model input
"""

from __future__ import annotations

import io
from typing import Final

import numpy as np
from PIL import Image

from settings import IMG_SIZE


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load an image from raw bytes."""
    return Image.open(io.BytesIO(image_bytes))


def resize_image(image: Image.Image, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Convert image to RGB, resize, and return as float32 NumPy array."""
    rgb = image.convert("RGB")
    resized = rgb.resize(size)
    return np.asarray(resized, dtype=np.float32)


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1]."""
    return image_array / 255.0


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Full preprocessing pipeline:
    - load from bytes
    - resize to IMG_SIZE
    - normalize to [0, 1]
    - add batch dimension
    Returns array of shape (1, H, W, 3).
    """
    image = load_image_from_bytes(image_bytes)
    image_array = resize_image(image, IMG_SIZE)
    normalized = normalize_image(image_array)
    return np.expand_dims(normalized, axis=0)

