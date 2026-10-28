"""Export trained artifacts to the static interactive dashboard."""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    CDC_CSV,
    DOCS_DATA,
    DOCS_IMG,
    DOCS_MODELS,
    FIGURES,
    M5_CALENDAR,
    M5_PRICES,
    M5_SALES,
    MODELS,
    ROOT,
    SEED,
    TABLES,
    UCI_CSV,
)
from .data_cdc import FEATURES as CDC_FEATURES, TARGET as CDC_TARGET, build_cdc_dataset
from .data_m5 import FEATURES as M5_FEATURES, build_m5_dataset
from .data_uci import build_uci_dataset
from .utils import load_json, save_json


def _tree_arrays(est, task: str) -> dict:
    t = est.tree_
    if task == "regression":
        v = t.value[:, 0, 0].astype(float)
    else:
        counts = t.value[:, 0, :].astype(float)
        v = counts[:, 1] / np.maximum(counts.sum(axis=1), 1e-12)
    return {
        "f": t.feature.astype(int).tolist(),
        "t": t.threshold.astype(float).tolist(),
        "l": t.children_left.astype(int).tolist(),
        "r": t.children_right.astype(int).tolist(),
        "v": v.tolist(),
    }


def _forest_predict(forest: dict, x: list[float]) -> tuple[float, float]:
    preds = []
    for tree in forest["trees"]:
        i = 0
        while tree["l"][i] >= 0:
            i = tree["l"][i] if x[tree["f"][i]] <= tree["t"][i] else tree["r"][i]
        preds.append(tree["v"][i])
    return float(np.mean(preds)), float(np.std(preds))


def _ada_predict(export: dict, x: list[float]) -> float:
    mean = np.asarray(export["scaler"]["mean"])
    scale = np.asarray(export["scaler"]["scale"])
    z = (np.asarray(x) - mean) / scale
    total = 0.0
    for tree, w in zip(export["trees"], export["weights"]):
        i = 0
        while tree["l"][i] >= 0:
            i = tree["l"][i] if z[tree["f"][i]] <= tree["t"][i] else tree["r"][i]
        total += w * (1.0 if tree["v"][i] >= 0.5 else -1.0)
    f = 2.0 * total / sum(export["weights"])
    return float(1.0 / (1.0 + np.exp(-f)))


def _cat_predict(export: dict, x: list[float]) -> np.ndarray:
    nc = len(export["classes"])
    raw = np.zeros(nc)
    for tree in export["trees"]:
        idx = 0
        for level, (fi, border) in enumerate(tree["splits"]):
            if x[fi] > border:
                idx |= (1 << level)
        raw += np.asarray(tree["leaves"][idx * nc:(idx + 1) * nc])
    raw = raw * export["scale"] + np.asarray(export["bias"])
    raw -= raw.max()
    e = np.exp(raw)
    return e / e.sum()


def export() -> dict:
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    DOCS_MODELS.mkdir(parents=True, exist_ok=True)
    DOCS_DATA.mkdir(parents=True, exist_ok=True)

    fig_map = {
        "m5_actual_vs_pred.png": "extra_actual_vs_pred.png",
        "m5_learning_curve.png": "extra_learning.png",
        "m5_importance.png": "extra_importance.png",
        "m5_pdp_2d.png": "extra_pdp2d.png",
        "m5_pdp_3d.png": "extra_pdp3d.png",
        "m5_series_forecast.png": "extra_series.png",
        "m5_search.png": "extra_search.png",
        "m5_residuals.png": "extra_residuals.png",
        "cdc_confusion.png": "ada_confusion.png",
        "cdc_roc.png": "ada_roc.png",
        "cdc_learning_curve.png": "ada_learning.png",
        "cdc_importance.png": "ada_importance.png",
        "cdc_thresholds.png": "ada_thresholds.png",
        "cdc_calibration.png": "ada_calibration.png",
        "cdc_decision_boundary.png": "ada_boundary.png",
        "cdc_search.png": "ada_search.png",
        "uci_confusion.png": "cat_confusion.png",
        "uci_learning_curve.png": "cat_learning.png",
        "uci_importance.png": "cat_importance.png",
        "uci_per_class.png": "cat_per_class.png",
        "uci_roc.png": "cat_roc.png",
        "uci_decision_boundary.png": "cat_boundary.png",
        "uci_calibration.png": "cat_calibration.png",
        "uci_search.png": "cat_search.png",
    }
    for src, dst in fig_map.items():
        p = FIGURES / src
        if p.exists():
            shutil.copy2(p, DOCS_IMG / dst)

    extra = load_json(TABLES / "extra_trees_results.json")
    ada = load_json(TABLES / "adaboost_results.json")
    cat = load_json(TABLES / "catboost_results.json")

    # ---------------- Extra Trees exported forest ----------------
    et_bundle = joblib.load(MODELS / "extra_trees_m5_demo.joblib")
    et_model = et_bundle["model"]
    et_encoder = et_bundle["encoder"]
    et_trees = [_tree_arrays(est, "regression") for est in et_model.estimators_]
    et_export = {
        "trees": et_trees,
        "feature_names": M5_FEATURES,
        "categorical": ["dept_id", "cat_id", "store_id", "state_id"],
        "cat_levels": extra["cat_levels"],
        "medians": extra["medians"],
        "quantiles": extra["quantiles"],
        "demo_metrics": extra["demo_metrics"],
        "n_estimators": len(et_trees),
    }
    # deterministic check against the sklearn demo model
    import sys
    sys.path.insert(0, str(ROOT))
    m5 = build_m5_dataset()
    df = m5["frame"]
    test = df[df["split"] == "test"].sample(5, random_state=SEED)
    enc = et_encoder.transform(test[M5_FEATURES])
    checks = []
    for row, pred in zip(enc, et_model.predict(enc)):
        p, _ = _forest_predict(et_export, row.tolist())
        checks.append(abs(p - float(pred)))
    et_export["inference_check_max_abs_diff"] = float(max(checks))
    save_json(et_export, DOCS_MODELS / "extra_trees.json")

    # ---------------- AdaBoost exported trees ----------------
    ada_bundle = joblib.load(MODELS / "adaboost_cdc.joblib")
    ada_pipe = ada_bundle["model"]
    ada_model = ada_pipe.named_steps["model"]
    scaler = ada_pipe.named_steps["scale"]
    ada_export = {
        "trees": [_tree_arrays(e, "binary") for e in ada_model.estimators_],
        "weights": ada_model.estimator_weights_.astype(float).tolist(),
        "scaler": {"mean": scaler.mean_.astype(float).tolist(),
                   "scale": scaler.scale_.astype(float).tolist()},
        "feature_names": ada_bundle["features"],
        "threshold": ada_bundle["threshold"],
        "defaults": ada["defaults"],
        "ranges": ada["ranges"],
        "metrics": ada["metrics"],
    }
    cdc = build_cdc_dataset()
    cdf = cdc["frame"]
    from sklearn.model_selection import train_test_split
    from .data_cdc import TARGET as CT
    Xc = cdf[CDC_FEATURES].sample(6, random_state=SEED)
    checks = []
    for _, row in Xc.iterrows():
        p = _ada_predict(ada_export, row.tolist())
        pp = float(ada_pipe.predict_proba(pd.DataFrame([row]))[0, 1])
        checks.append(abs(p - pp))
    ada_export["inference_check_max_abs_diff"] = float(max(checks))
    save_json(ada_export, DOCS_MODELS / "adaboost.json")

    # ---------------- CatBoost exported oblivious trees ----------------
    cb_bundle = joblib.load(MODELS / "catboost_uci_demo.joblib")
    cb_model = cb_bundle["model"]
    cb_cols = cb_bundle["columns"]
    tmp = Path(tempfile.gettempdir()) / "catboost_export.json"
    cb_model.save_model(str(tmp), format="json")
    with open(tmp, encoding="utf-8") as fh:
        cb_json = json.load(fh)
    cb_trees = []
    for t in cb_json["oblivious_trees"]:
        splits = [[int(s["float_feature_index"]), float(s["border"])] for s in t["splits"]]
        cb_trees.append({"splits": splits, "leaves": [float(v) for v in t["leaf_values"]]})
    scale, bias = cb_json["scale_and_bias"]
    cb_export = {
        "trees": cb_trees,
        "scale": float(scale),
        "bias": [float(b) for b in bias],
        "classes": cb_bundle["classes"],
        "columns": cb_cols,
        "feature_names": cat["feature_order"],
        "categorical": cat["categorical"],
        "numeric": cat["numeric"],
        "cat_levels": cat["cat_levels"],
        "defaults": cat["defaults"],
        "ranges": cat["ranges"],
        "demo_metrics": cat["demo_metrics"],
        "n_trees": len(cb_trees),
    }
    # build one-hot sample exactly as pandas did
    uci = build_uci_dataset()
    udf = uci["frame"]
    Xtr, Xte, ytr, yte = train_test_split(udf[uci["features"]], udf["Target"],
                                          test_size=0.20, stratify=udf["Target"], random_state=SEED)
    all_cat = pd.get_dummies(pd.concat([Xtr, Xte])[uci["features"]],
                             columns=uci["categorical"], dtype=float)
    Xte_oh = all_cat.iloc[len(Xtr):].reset_index(drop=True)
    checks = []
    for i in range(6):
        p = _cat_predict(cb_export, Xte_oh.iloc[i].tolist())
        pp = cb_model.predict_proba(Xte_oh.iloc[[i]])[0]
        checks.append(float(np.abs(p - pp).max()))
    cb_export["inference_check_max_abs_diff"] = float(max(checks))
    save_json(cb_export, DOCS_MODELS / "catboost.json")

    # ---------------- dashboard payload ----------------
    dashboard = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "project": {
            "title": "Árboles y Ensambles · Extra Trees, AdaBoost y CatBoost",
            "subtitle": "Tres casos de uso reales: retail, salud pública y abandono universitario",
            "course": "Aprendizaje Supervisado · Familia 1: Árboles y Ensambles",
            "team": "Equipo 2",
            "author": "Luchi-Marita",
        },
        "summary": {
            "extra_trees": {
                "headline": "RMSE 1.998 · R² 0.607",
                "metric_label": "Error cuadrático medio",
                "value": extra["metrics"]["rmse"], "r2": extra["metrics"]["r2"],
                "dataset": "Walmart M5", "n": extra["summary"]["n_rows"],
            },
            "adaboost": {
                "headline": "ROC-AUC 0.821 · F1 0.758",
                "metric_label": "ROC-AUC",
                "value": ada["metrics"]["roc_auc"], "f1": ada["metrics"]["f1"],
                "dataset": "CDC BRFSS 2015", "n": ada["summary"]["n_rows"],
            },
            "catboost": {
                "headline": "Macro-F1 0.693 · AUC-OvR 0.886",
                "metric_label": "F1 macro",
                "value": cat["metrics"]["f1_macro"], "auc": cat["metrics"]["roc_auc_ovr"],
                "dataset": "UCI Dropout", "n": cat["summary"]["n_rows"],
            },
        },
        "models": {"extra_trees": extra, "adaboost": ada, "catboost": cat},
        "figures": fig_map,
        "inference_checks": {
            "extra_trees": et_export["inference_check_max_abs_diff"],
            "adaboost": ada_export["inference_check_max_abs_diff"],
            "catboost": cb_export["inference_check_max_abs_diff"],
        },
    }
    save_json(dashboard, DOCS_DATA / "dashboard.json")

    print("exported models:", {k: v["inference_check_max_abs_diff"] for k, v in
                               [("extra", et_export), ("ada", ada_export), ("cat", cb_export)]})
    for p in sorted(DOCS_MODELS.glob("*.json")):
        print(p.name, round(p.stat().st_size / 1e6, 2), "MB")
    print("dashboard:", round((DOCS_DATA / "dashboard.json").stat().st_size / 1e6, 2), "MB")
    return dashboard


if __name__ == "__main__":
    export()
