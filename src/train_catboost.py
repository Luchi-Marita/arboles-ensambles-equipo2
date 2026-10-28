"""CatBoost experiment: UCI student dropout / academic success."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from .config import FIGURES, MODELS, SEED, TABLES
from .data_uci import CATEGORICAL, NUMERIC, TARGET, TARGET_ORDER, build_uci_dataset
from .metrics import dropout_binary_metrics, multiclass_metrics
from .plot_style import AMBER, BLUE, GRID, INK, MUTED, PALETTE, ROSE, TEAL, VIOLET, savefig, setup
from .utils import save_json, timed


def _model(iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
           random_strength=1.0, bagging_temperature=0.5,
           cat_idx=None, auto_class_weights=None, seed=SEED):
    return CatBoostClassifier(
        iterations=int(iterations), depth=int(depth), learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg), random_strength=float(random_strength),
        bagging_temperature=float(bagging_temperature),
        cat_features=tuple(cat_idx) if cat_idx is not None else None,
        auto_class_weights=auto_class_weights,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=seed, verbose=0, allow_writing_files=False, thread_count=-1,
    )


def run() -> dict:
    setup()
    t0 = time.time()
    data = build_uci_dataset()
    df = data["frame"]
    X, y = df[data["features"]], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL]

    base = _model(cat_idx=cat_idx)
    space = {
        "iterations": [250, 350, 500, 650],
        "depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "l2_leaf_reg": [1, 3, 5, 10],
        "random_strength": [0, 0.5, 1, 2],
        "bagging_temperature": [0, 0.5, 1],
        "auto_class_weights": [None, "Balanced"],
    }
    search = RandomizedSearchCV(
        base, space, n_iter=14, scoring="f1_macro",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=1, refit=False, verbose=1,
    )
    with timed("CatBoost RandomizedSearchCV"):
        search.fit(Xtr, ytr)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    print("best:", best, "cv macro-F1:", search.best_score_)

    Xa, Xc, ya, yc = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    eval_pool = Pool(Xc, yc, cat_features=cat_idx)
    final = _model(cat_idx=cat_idx, **best)
    with timed("CatBoost final fit"):
        final.fit(Xa, ya, eval_set=eval_pool, use_best_model=True,
                  early_stopping_rounds=80, verbose=False)
    print("trees used:", final.tree_count_)

    classes = [str(c) for c in final.classes_]
    proba = final.predict_proba(Xte)
    metrics = multiclass_metrics(yte.to_numpy(), proba, classes)
    metrics["dropout_binary"] = dropout_binary_metrics(yte.to_numpy(), proba, classes)
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    print("classes:", classes, "per class:", metrics["per_class"])

    rng = np.random.default_rng(SEED + 11)
    accs, f1s = [], []
    yte_arr = yte.to_numpy()
    for _ in range(400):
        idx = rng.integers(0, len(yte_arr), len(yte_arr))
        m = multiclass_metrics(yte_arr[idx], proba[idx], classes)
        accs.append(m["accuracy"]); f1s.append(m["f1_macro"])
    metrics["bootstrap"] = {
        "accuracy": {"lo": float(np.percentile(accs, 2.5)), "hi": float(np.percentile(accs, 97.5))},
        "f1_macro": {"lo": float(np.percentile(f1s, 2.5)), "hi": float(np.percentile(f1s, 97.5))},
    }

    # learning dynamics on the test set
    n_trees = final.tree_count_
    grid = sorted(set([min(n_trees, n) for n in range(50, n_trees + 1, 50)] + [n_trees]))
    hist = {"trees": [], "accuracy": [], "f1_macro": [], "log_loss": []}
    for n in grid:
        p = final.predict_proba(Xte, ntree_end=n)
        m = multiclass_metrics(yte_arr, p, classes)
        hist["trees"].append(int(n)); hist["accuracy"].append(m["accuracy"])
        hist["f1_macro"].append(m["f1_macro"]); hist["log_loss"].append(m["log_loss"])

    imp = pd.DataFrame({
        "feature": data["features"],
        "importance": final.get_feature_importance(),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(TABLES / "uci_feature_importance.csv", index=False)

    # per-class ROC (one-vs-rest)
    roc_curves = {}
    for k, c in enumerate(classes):
        fpr, tpr, _ = roc_curve((yte_arr == c).astype(int), proba[:, k])
        keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 180)).astype(int)
        roc_curves[c] = {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist(),
                         "auc": float(roc_auc_score((yte_arr == c).astype(int), proba[:, k]))}

    # 2D dropout-risk surface over the two most important numeric variables
    num_sorted = [f for f in imp["feature"] if f in NUMERIC]
    xcol, ycol = num_sorted[0], num_sorted[1]
    ref = Xtr.iloc[0].copy()
    for c in CATEGORICAL:
        ref[c] = Xtr[c].mode().iloc[0]
    for c in NUMERIC:
        ref[c] = float(Xtr[c].median())
    q1, q99 = Xtr[num_sorted].quantile(0.02), Xtr[num_sorted].quantile(0.98)
    gx = np.linspace(q1[xcol], q99[xcol], 55)
    gy = np.linspace(q1[ycol], q99[ycol], 55)
    grid_df = pd.DataFrame([ref] * (len(gx) * len(gy)))
    grid_df[xcol] = np.repeat(gx, len(gy))
    grid_df[ycol] = np.tile(gy, len(gx))
    for c in CATEGORICAL:
        grid_df[c] = grid_df[c].astype(str)
    for c in NUMERIC:
        grid_df[c] = grid_df[c].astype(float)
    pg = final.predict_proba(grid_df)
    didx = classes.index("Dropout")
    zz = pg[:, didx].reshape(len(gx), len(gy))

    rng = np.random.default_rng(SEED + 13)
    sidx = rng.choice(len(Xte), min(500, len(Xte)), replace=False)
    scatter = []
    for i in sidx:
        row = Xte.iloc[i]
        scatter.append({"x": float(row[xcol]), "y": float(row[ycol]),
                        "actual": str(yte.iloc[i])})

    # calibration of the Dropout class
    frac_pos, mean_pred = calibration_curve((yte_arr == "Dropout").astype(int),
                                            proba[:, didx], n_bins=10, strategy="quantile")

    # test sample for the dashboard
    test_sample = []
    for i in rng.choice(len(Xte), min(600, len(Xte)), replace=False):
        row = Xte.iloc[i]
        test_sample.append({
            "actual": str(yte.iloc[i]),
            "probs": {c: float(proba[i, k]) for k, c in enumerate(classes)},
            "approved1": float(row["Curricular_units_1st_sem_approved"]),
            "approved2": float(row["Curricular_units_2nd_sem_approved"]),
            "grade1": float(row["Curricular_units_1st_sem_grade"]),
            "grade2": float(row["Curricular_units_2nd_sem_grade"]),
            "tuition": str(row["Tuition_fees_up_to_date"]),
            "scholarship": str(row["Scholarship_holder"]),
            "age": float(row["Age_at_enrollment"]),
            "course": str(row["Course"]),
        })

    # ---- figures -------------------------------------------------------
    cm = np.array(metrics["confusion"])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap="Purples")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK,
                    fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=20)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión · CatBoost (test)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.72)
    savefig(fig, FIGURES / "uci_confusion.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(hist["trees"], hist["f1_macro"], color=VIOLET, lw=2.2, label="F1 macro")
    ax.plot(hist["trees"], hist["accuracy"], color=TEAL, lw=1.8, label="Accuracy")
    ax2 = ax.twinx()
    ax2.plot(hist["trees"], hist["log_loss"], color=AMBER, lw=1.6, ls="--", label="Log-loss")
    ax2.set_ylabel("Log-loss", color=AMBER); ax2.tick_params(axis="y", colors=AMBER)
    ax2.grid(False)
    ax.set_xlabel("Iteraciones (árboles oblivios)")
    ax.set_ylabel("Métrica")
    ax.set_title("Curva de aprendizaje · CatBoost")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)
    savefig(fig, FIGURES / "uci_learning_curve.png")

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    top = imp.head(16).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color=VIOLET, height=0.68)
    ax.set_xlabel("PredictionValuesChange (importancia)")
    ax.set_title("CatBoost · importancia de variables")
    savefig(fig, FIGURES / "uci_importance.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = list(metrics["per_class"].keys())
    xx = np.arange(len(labels)); w = 0.26
    for k, (metric, col) in enumerate([("precision", BLUE), ("recall", ROSE), ("f1", TEAL)]):
        vals = [metrics["per_class"][c][metric] for c in labels]
        ax.bar(xx + (k - 1) * w, vals, width=w, color=col, label=metric.capitalize())
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.03); ax.set_ylabel("valor")
    ax.set_title("Desempeño por clase · CatBoost")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    savefig(fig, FIGURES / "uci_per_class.png")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for c, col in zip(classes, [ROSE, AMBER, TEAL]):
        ax.plot(roc_curves[c]["fpr"], roc_curves[c]["tpr"], color=col, lw=1.9,
                label=f"{c} (AUC={metrics['per_class'][c]['f1']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
    ax.set_title("Curvas ROC one-vs-rest · CatBoost")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, FIGURES / "uci_roc.png")

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    cs = ax.contourf(gx, gy, zz.T, levels=np.linspace(zz.min(), zz.max(), 20), cmap="magma_r")
    ax.contour(gx, gy, zz.T, levels=[0.33, 0.5], colors="white", linewidths=1.0)
    for lab, col in [("Dropout", "#f59e0b"), ("Enrolled", "#38bdf8"), ("Graduate", "#a3e635")]:
        pts = [s for s in scatter if s["actual"] == lab]
        if pts:
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=14,
                       c=col, edgecolor="black", linewidth=0.25, alpha=0.8, label=lab)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title("Superficie de riesgo de abandono (P(Dropout))")
    fig.colorbar(cs, ax=ax, label="P(Dropout)")
    ax.legend(loc="best", fontsize=8)
    savefig(fig, FIGURES / "uci_decision_boundary.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(mean_pred, frac_pos, color=ROSE, marker="o", ms=4, lw=1.6, label="Dropout")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración · clase Dropout")
    ax.legend()
    savefig(fig, FIGURES / "uci_calibration.png")

    # ---- demo model for the browser (one-hot encoded, float-only) ------
    all_cat = pd.get_dummies(pd.concat([Xtr, Xte], axis=0)[data["features"]],
                             columns=CATEGORICAL, dtype=float)
    onehot_cols = list(all_cat.columns)
    Xtr_oh = all_cat.iloc[:len(Xtr)]
    Xte_oh = all_cat.iloc[len(Xtr):]
    demo_params = {k: v for k, v in best.items() if k != "iterations"}
    demo = _model(iterations=min(400, int(best["iterations"])), cat_idx=None,
                  **demo_params)
    Xd, Xdv, yd, ydv = train_test_split(Xtr_oh, ytr, test_size=0.15,
                                        stratify=ytr, random_state=SEED + 1)
    demo.fit(Xd, yd, eval_set=(Xdv, ydv), use_best_model=True,
             early_stopping_rounds=60, verbose=False)
    demo_proba = demo.predict_proba(Xte_oh)
    demo_metrics = multiclass_metrics(yte.to_numpy(), demo_proba, [str(c) for c in demo.classes_])
    print("demo:", {k: round(v, 4) for k, v in demo_metrics.items() if isinstance(v, float)})

    joblib.dump({"model": final, "features": data["features"], "categorical": CATEGORICAL,
                 "classes": classes}, MODELS / "catboost_uci.joblib")
    joblib.dump({"model": demo, "columns": onehot_cols, "classes": [str(c) for c in demo.classes_]},
                MODELS / "catboost_uci_demo.joblib")

    search_df = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    search_df.insert(0, "rank", np.arange(1, len(search_df) + 1))
    search_df["config"] = search_df["params"].apply(
        lambda p: " | ".join(f"{k}={v}" for k, v in sorted(p.items())))
    search_df[["rank", "mean_test_score", "std_test_score", "config"]].to_csv(
        TABLES / "uci_search_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    topn = search_df.head(8).iloc[::-1]
    ax.barh([f"#{i+1}" for i in range(len(topn))], topn["mean_test_score"], color=VIOLET, height=0.66)
    ax.set_xlabel("F1 macro de validación cruzada")
    ax.set_title("RandomizedSearchCV · CatBoost (14 configuraciones)")
    savefig(fig, FIGURES / "uci_search.png")

    cat_levels = {c: [str(v) for v in sorted(df[c].astype(str).unique())] for c in CATEGORICAL}
    result = {
        "model": "CatBoost",
        "task": "multiclass_classification",
        "dataset": "UCI · Predict Students' Dropout and Academic Success",
        "summary": data["summary"],
        "metrics": metrics,
        "demo_metrics": demo_metrics,
        "best_params": best,
        "cv_f1_macro": float(search.best_score_),
        "classes": classes,
        "tree_count": int(final.tree_count_),
        "search": search_df[["rank", "mean_test_score", "std_test_score", "config"]].head(14).to_dict("records"),
        "learning": hist,
        "roc": roc_curves,
        "importance": imp.head(20).to_dict("records"),
        "decision": {"x": gx.tolist(), "y": gy.tolist(), "z": zz.tolist(),
                     "x_label": xcol, "y_label": ycol, "scatter": scatter},
        "calibration": {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()},
        "test_sample": test_sample,
        "defaults": {c: (Xtr[c].mode().iloc[0] if c in CATEGORICAL else float(Xtr[c].median()))
                     for c in data["features"]},
        "ranges": {c: {"min": float(Xtr[c].min()), "max": float(Xtr[c].max()),
                       "p05": float(Xtr[c].quantile(0.05)), "p95": float(Xtr[c].quantile(0.95))}
                   for c in NUMERIC},
        "cat_levels": cat_levels,
        "feature_order": data["features"],
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(result, TABLES / "catboost_results.json")
    print(f"CatBoost finished in {result['runtime_seconds']}s")
    return result


if __name__ == "__main__":
    run()
