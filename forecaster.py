"""Run the promoted model, with or without PyTorch.

models/current.json names the live model file:
* ``*.pt``   - PyTorch checkpoint (the training PC; uses the GPU if there is one)
* ``*.onnx`` - exported model run by OpenCV DNN, no PyTorch needed (benh-pi)

This module must not import torch at top level: the Pi has no PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config


def current_model_info() -> dict | None:
    try:
        return json.loads(config.CURRENT_MODEL_FILE.read_text())
    except (OSError, ValueError):
        return None


def current_model_path() -> Path | None:
    """Path of the promoted model from models/current.json, or None."""
    info = current_model_info()
    if not info or "file" not in info:
        return None
    p = config.MODELS_DIR / info["file"]
    return p if p.exists() else None


class OnnxForecaster:
    """OpenCV DNN runner for an exported model (works on 32-bit ARM)."""

    def __init__(self, path: Path):
        import cv2
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.device = "cpu (OpenCV DNN)"

    def predict(self, inputs_mm: np.ndarray) -> np.ndarray:
        """(n_in, G, G) mm/h -> (n_out, G, G) mm/h."""
        x = np.log1p(np.clip(inputs_mm, 0, None)).astype(np.float32)[None]
        self.net.setInput(x)
        y = self.net.forward()
        return np.clip(np.expm1(y[0]), 0, None).astype(np.float32)


class TorchForecaster:
    def __init__(self, path: Path):
        from model import device, load_checkpoint
        self.dev = device()
        self.model, self.meta = load_checkpoint(path, self.dev)
        self.device = str(self.dev)

    def predict(self, inputs_mm: np.ndarray) -> np.ndarray:
        from model import predict_mm
        return predict_mm(self.model, inputs_mm, self.dev)


def load_forecaster(path: Path):
    """(forecaster, grid) for a .pt or .onnx model file."""
    path = Path(path)
    if path.suffix == ".onnx":
        info = current_model_info() or {}
        return OnnxForecaster(path), int(info.get("grid", config.GRID))
    f = TorchForecaster(path)
    return f, f.meta.get("grid")
