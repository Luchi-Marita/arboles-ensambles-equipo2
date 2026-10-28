"""Metric helpers for regression and classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100),
        "smape": float(np.mean(2 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)) * 100),
        "wape": float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1e-8) * 100),
        "bias": float(np.mean(err)),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "n": int(len(y_true)),
    }


def bootstrap_regression(y_true, y_pred, n_boot: int = 600, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    vals = {"rmse": [], "mae": [], "r2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        e = yp - yt
        vals["rmse"].append(np.sqrt(np.mean(e ** 2)))
        vals["mae"].append(np.mean(np.abs(e)))
        denom = np.sum((yt - yt.mean()) ** 2)
        vals["r2"].append(1 - np.sum(e ** 2) / denom if denom > 0 else np.nan)
    out = {}
    for k, v in vals.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        out[k] = {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                  "mean": float(v.mean())}
    return out


def binary_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def multiclass_metrics(y_true, proba, classes: list[str], threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, pred, labels=classes)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, pred, labels=classes, zero_division=0)
    per_class = {
        c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
            "support": int(sup[i])}
        for i, c in enumerate(classes)
    }
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", labels=classes, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", labels=classes, zero_division=0)),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "roc_auc_ovr": auc,
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
        "confusion": cm.tolist(),
        "per_class": per_class,
        "n": int(len(y_true)),
    }


def dropout_binary_metrics(y_true, proba, classes: list[str]) -> dict:
    """One-vs-rest metrics for the 'Dropout' class (risk screening)."""
    y_true = np.asarray(y_true)
    idx = list(classes).index("Dropout")
    y_bin = (y_true == "Dropout").astype(int)
    p = np.asarray(proba)[:, idx]
    out = binary_metrics(y_bin, p, threshold=0.5)
    out["label"] = "Dropout vs resto"
    return out


def full_classification_report(y_true, pred, labels) -> str:
    return classification_report(y_true, pred, labels=labels, zero_division=0)
