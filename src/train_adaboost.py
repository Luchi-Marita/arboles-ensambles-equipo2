"""AdaBoost experiment: CDC BRFSS diabetes classification."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform

from .config import FIGURES, MODELS, SEED, TABLES
from .data_cdc import FEATURES, TARGET, build_cdc_dataset
from .metrics import binary_metrics
from .plot_style import AMBER, BLUE, GRID, INK, MUTED, ROSE, TEAL, savefig, setup
from .utils import save_json, timed


def _make_model(n_estimators=200, learning_rate=0.1, depth=2, min_leaf=1):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                             random_state=SEED),
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=SEED,
        )),
    ])


def _best_f1_threshold(y, p):
    grid = np.linspace(0.10, 0.90, 161)
    best = (0.5, -1)
    for t in grid:
        pred = (p >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def run() -> dict:
    setup()
    t0 = time.time()
    data = build_cdc_dataset()
    df = data["frame"]
    X, y = df[FEATURES], df[TARGET].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    base = _make_model()
    space = {
        "model__n_estimators": randint(100, 420),
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.4],
        "model__estimator__max_depth": [1, 2, 3, 4],
        "model__estimator__min_samples_leaf": [1, 5, 10, 20],
    }
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(Xtr), min(35_000, len(Xtr)), replace=False)
    search = RandomizedSearchCV(
        base, space, n_iter=14, scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=-1, refit=False, verbose=1,
    )
    with timed("AdaBoost RandomizedSearchCV"):
        search.fit(Xtr.iloc[sub_idx], ytr[sub_idx])
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    print("best:", best, "cv AUC:", search.best_score_)

    # threshold selected on a held-out slice of the training data
    Xa, Xc, ya, yc = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    thr_model = _make_model(
        n_estimators=best["model__n_estimators"],
        learning_rate=best["model__learning_rate"],
        depth=best["model__estimator__max_depth"],
        min_leaf=best["model__estimator__min_samples_leaf"],
    ).fit(Xa, ya)
    p_cal = thr_model.predict_proba(Xc)[:, 1]
    threshold, f1_cal = _best_f1_threshold(yc, p_cal)
    print(f"selected threshold={threshold:.3f} (calibration F1={f1_cal:.4f})")

    final = _make_model(
        n_estimators=best["model__n_estimators"],
        learning_rate=best["model__learning_rate"],
        depth=best["model__estimator__max_depth"],
        min_leaf=best["model__estimator__min_samples_leaf"],
    )
    with timed("AdaBoost final fit"):
        final.fit(Xtr, ytr)
    proba = final.predict_proba(Xte)[:, 1]
    metrics = binary_metrics(yte, proba, 0.5)
    metrics["tuned"] = binary_metrics(yte, proba, threshold)
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    # bootstrap CI for AUC / F1
    rng = np.random.default_rng(SEED + 7)
    aucs, f1s = [], []
    for _ in range(400):
        idx = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[idx])) < 2:
            continue
        m = binary_metrics(yte[idx], proba[idx], threshold)
        aucs.append(m["roc_auc"]); f1s.append(m["f1"])
    metrics["bootstrap"] = {
        "roc_auc": {"lo": float(np.percentile(aucs, 2.5)), "hi": float(np.percentile(aucs, 97.5))},
        "f1": {"lo": float(np.percentile(f1s, 2.5)), "hi": float(np.percentile(f1s, 97.5))},
    }

    # staged learning dynamics
    stages, auc_hist, f1_hist, acc_hist = [], [], [], []
    Xte_scaled = final.named_steps["scale"].transform(Xte)
    for i, p in enumerate(final.named_steps["model"].staged_predict_proba(Xte_scaled)):
        if (i + 1) % 10 == 0 or i == 0:
            m = binary_metrics(yte, p[:, 1], threshold)
            stages.append(i + 1); auc_hist.append(m["roc_auc"])
            f1_hist.append(m["f1"]); acc_hist.append(m["accuracy"])

    fpr, tpr, _ = roc_curve(yte, proba)
    prec, rec, _ = precision_recall_curve(yte, proba)
    n_pts = min(len(fpr), 220)
    keep = np.linspace(0, len(fpr) - 1, n_pts).astype(int)
    roc_web = {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()}
    keep = np.linspace(0, len(prec) - 1, n_pts).astype(int)
    pr_web = {"precision": prec[keep].tolist(), "recall": rec[keep].tolist()}

    # threshold sweep
    grid = np.linspace(0.05, 0.95, 91)
    sweep = {"threshold": grid.tolist(), "f1": [], "precision": [], "recall": [], "accuracy": []}
    for t in grid:
        m = binary_metrics(yte, proba, float(t))
        sweep["f1"].append(m["f1"]); sweep["precision"].append(m["precision"])
        sweep["recall"].append(m["recall"]); sweep["accuracy"].append(m["accuracy"])

    imp = pd.DataFrame({
        "feature": FEATURES, "importance": final.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(TABLES / "cdc_feature_importance.csv", index=False)

    try:
        pi = permutation_importance(
            final, Xte.iloc[:2000].to_numpy(), yte[:2000], n_repeats=2,
            scoring="roc_auc", random_state=SEED, n_jobs=-1,
        )
        perm = pd.DataFrame({
            "feature": FEATURES, "perm_importance": pi.importances_mean,
            "perm_std": pi.importances_std,
        }).sort_values("perm_importance", ascending=False).reset_index(drop=True)
        perm.to_csv(TABLES / "cdc_permutation_importance.csv", index=False)
    except Exception as exc:  # pragma: no cover
        print("permutation importance skipped:", exc)
        perm = imp.copy()

    # calibration
    frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=12, strategy="quantile")

    # 2D decision surface on the two most important features
    top2 = imp["feature"].head(2).tolist()
    xcol, ycol = top2[0], top2[1]
    med = Xtr.median(numeric_only=True)
    q1, q99 = Xtr.quantile(0.01), Xtr.quantile(0.99)
    gx = np.linspace(q1[xcol], q99[xcol], 76)
    gy = np.linspace(q1[ycol], q99[ycol], 76)
    grid_df = pd.DataFrame([med.values] * (len(gx) * len(gy)), columns=FEATURES)
    grid_df[xcol] = np.repeat(gx, len(gy))
    grid_df[ycol] = np.tile(gy, len(gx))
    zz = final.predict_proba(grid_df)[:, 1].reshape(len(gx), len(gy))

    rng = np.random.default_rng(SEED + 2)
    sidx = rng.choice(len(Xte), min(700, len(Xte)), replace=False)
    scatter = [{
        "x": float(Xte.iloc[i][xcol]), "y": float(Xte.iloc[i][ycol]),
        "actual": int(yte[i]),
    } for i in sidx]

    test_sample = [{
        "prob": float(proba[i]), "actual": int(yte[i]),
        "bmi": float(Xte.iloc[i]["BMI"]), "age": float(Xte.iloc[i]["Age"]),
        "genhlth": float(Xte.iloc[i]["GenHlth"]), "comorbidity": float(Xte.iloc[i]["Comorbidity"]),
    } for i in rng.choice(len(Xte), min(700, len(Xte)), replace=False)]

    # ---- figures -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    cm = np.array(metrics["tuned"]["confusion"])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No diabetes", "Diabetes"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No diabetes", "Diabetes"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión · umbral {threshold:.2f}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75)
    savefig(fig, FIGURES / "cdc_confusion.png")

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(fpr, tpr, color=BLUE, lw=2.3, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC · AdaBoost (CDC)")
    ax.legend(loc="lower right")
    savefig(fig, FIGURES / "cdc_roc.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(stages, auc_hist, color=BLUE, lw=2.1, label="ROC-AUC")
    ax.plot(stages, f1_hist, color=AMBER, lw=1.9, label="F1")
    ax.plot(stages, acc_hist, color=TEAL, lw=1.7, label="Accuracy")
    ax.set_xlabel("Número de estimadores débiles")
    ax.set_ylabel("Métrica")
    ax.set_title("Curva de aprendizaje · AdaBoost")
    ax.legend()
    savefig(fig, FIGURES / "cdc_learning_curve.png")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    top = imp.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color=TEAL, height=0.68)
    ax.set_xlabel("Importancia (impureza ponderada por α)")
    ax.set_title("AdaBoost · predictores más relevantes")
    savefig(fig, FIGURES / "cdc_importance.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(sweep["threshold"], sweep["f1"], color=BLUE, lw=2, label="F1")
    ax.plot(sweep["threshold"], sweep["recall"], color=ROSE, lw=1.7, label="Recall")
    ax.plot(sweep["threshold"], sweep["precision"], color=TEAL, lw=1.7, label="Precisión")
    ax.axvline(threshold, color=AMBER, ls="--", lw=1.3, label=f"umbral elegido = {threshold:.2f}")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Métrica")
    ax.set_title("Compromiso sensibilidad/precisión en cribado")
    ax.legend(fontsize=8)
    savefig(fig, FIGURES / "cdc_thresholds.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(mean_pred, frac_pos, color=BLUE, marker="o", ms=4, lw=1.6, label="AdaBoost")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="calibración perfecta")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Curva de calibración (test)")
    ax.legend()
    savefig(fig, FIGURES / "cdc_calibration.png")

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    cs = ax.contourf(gx, gy, zz.T, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.9)
    ax.contour(gx, gy, zz.T, levels=[0.5], colors=INK, linewidths=1.5)
    for lab, col in [(0, "#1d4ed8"), (1, "#b91c1c")]:
        pts = [s for s in scatter if s["actual"] == lab]
        if pts:
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=13, c=col,
                       edgecolor="white", linewidth=0.25, alpha=0.8,
                       label="No diabetes" if lab == 0 else "Diabetes")
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title("Frontera de decisión 2D (perfil mediano)")
    fig.colorbar(cs, ax=ax, label="P(diabetes)")
    ax.legend(loc="best", fontsize=8)
    savefig(fig, FIGURES / "cdc_decision_boundary.png")

    # ---- artifacts -----------------------------------------------------
    joblib.dump({"model": final, "features": FEATURES, "threshold": threshold},
                MODELS / "adaboost_cdc.joblib")
    search_df = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    search_df.insert(0, "rank", np.arange(1, len(search_df) + 1))
    search_df["config"] = search_df["params"].apply(
        lambda p: " | ".join(f"{k.replace('model__', '').replace('estimator__', '')}={v}"
                             for k, v in sorted(p.items())))
    search_df[["rank", "mean_test_score", "std_test_score", "config"]].to_csv(
        TABLES / "cdc_search_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    topn = search_df.head(8).iloc[::-1]
    ax.barh([f"#{i+1}" for i in range(len(topn))], topn["mean_test_score"], color=TEAL, height=0.66)
    ax.set_xlabel("ROC-AUC de validación cruzada")
    ax.set_title("RandomizedSearchCV · AdaBoost (14 configuraciones)")
    savefig(fig, FIGURES / "cdc_search.png")

    result = {
        "model": "AdaBoost",
        "task": "binary_classification",
        "dataset": "CDC BRFSS 2015 · Diabetes Health Indicators",
        "summary": data["summary"],
        "metrics": metrics,
        "threshold": threshold,
        "best_params": {k.replace("model__", "").replace("estimator__", ""): v
                        for k, v in best.items()},
        "cv_auc": float(search.best_score_),
        "search": search_df[["rank", "mean_test_score", "std_test_score", "config"]].head(14).to_dict("records"),
        "learning": {"estimators": stages, "roc_auc": auc_hist, "f1": f1_hist, "accuracy": acc_hist},
        "roc": roc_web, "pr": pr_web, "threshold_sweep": sweep,
        "importance": imp.head(20).to_dict("records"),
        "permutation": perm.head(20).to_dict("records"),
        "calibration": {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()},
        "decision": {"x": gx.tolist(), "y": gy.tolist(), "z": zz.tolist(),
                     "x_label": xcol, "y_label": ycol, "scatter": scatter},
        "test_sample": test_sample,
        "defaults": Xtr.median(numeric_only=True).to_dict(),
        "ranges": {f: {"min": float(Xtr[f].min()), "max": float(Xtr[f].max()),
                       "p05": float(Xtr[f].quantile(0.05)), "p95": float(Xtr[f].quantile(0.95))}
                   for f in FEATURES},
        "feature_order": FEATURES,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(result, TABLES / "adaboost_results.json")
    print(f"AdaBoost finished in {result['runtime_seconds']}s")
    return result


if __name__ == "__main__":
    run()
