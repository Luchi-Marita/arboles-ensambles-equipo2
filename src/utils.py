"""Small shared helpers."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _sanitize(obj: Any, _depth: int = 0) -> Any:
    if _depth > 40:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, _depth + 1) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist(), _depth + 1)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize(obj), fh, indent=2, ensure_ascii=False,
                  default=json_default, allow_nan=False)


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def timed(label: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"[start] {label}", flush=True)
            return self

        def __exit__(self, *exc):
            print(f"[done ] {label}  ({time.perf_counter() - self.t0:.1f}s)", flush=True)

    return _Timer()


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)
