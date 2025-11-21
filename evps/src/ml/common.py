import json
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - tf may be optional for some environments
    tf = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


for _path in [PROCESSED_DIR, SCALERS_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(_path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set seeds across random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
