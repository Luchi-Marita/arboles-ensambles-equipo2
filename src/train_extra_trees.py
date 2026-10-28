"""Extra Trees experiment: Walmart M5 daily unit-sales regression."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import randint, uniform

from .config import FIGURES, M5_SEARCH_ROWS, MODELS, SEED, TABLES
from .data_m5 import CATEGORICAL, FEATURES, TARGET, build_m5_dataset, split_xy
from .metrics import bootstrap_regression, regression_metrics
from .plot_style import AMBER, BLUE, INK, MUTED, PALETTE, ROSE, TEAL, savefig, setup
from .utils import mape, save_json, smape, timed


def _flatten_params(params: dict) -> dict:
    return {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
            for k, v in params.items()}


def _encode(train, val, test):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(pd.concat([train, val], axis=0))
    cols = FEATURES
    return tuple(pd.DataFrame(enc.transform(x), columns=cols, index=x.index)
                 for x in (train, val, test)), enc


def run() -> dict:
    setup()
    t0 = time.time()
    data = build_m5_dataset()
    df = data["frame"]
    summary = data["summary"]
    parts = split_xy(df, FEATURES)
    Xtr, ytr, mtr = parts["train"]
    Xval, yval, mval = parts["val"]
    Xte, yte, mte = parts["test"]
    (Xtr_e, Xval_e, Xte_e), encoder = _encode(Xtr, Xval, Xte)

    print(f"M5 rows: train={len(Xtr):,} val={len(Xval):,} test={len(Xte):,}")

    base = ExtraTreesRegressor(random_state=SEED, n_jobs=-1)
    space = {
        "n_estimators": randint(200, 380),
        "max_depth": [20, 28, 36],
        "max_features": ["sqrt", 0.5, 0.7, 0.9],
        "min_samples_leaf": [4, 8, 12, 20],
        "min_samples_split": [2, 10, 20],
        "bootstrap": [False, True],
    }
    rng = np.random.default_rng(SEED)
    n_search = min(M5_SEARCH_ROWS, len(Xtr_e))
    idx = rng.choice(len(Xtr_e), n_search, replace=False)
    Xs, ys = Xtr_e.iloc[idx], ytr[idx]
    Xsearch = pd.concat([Xs, Xval_e], axis=0)
    ysearch = np.concatenate([ys, yval])
    cv = [(np.arange(len(Xs)), np.arange(len(Xs), len(Xs) + len(Xval_e)))]

    search = RandomizedSearchCV(
        base, space, n_iter=12, scoring="neg_root_mean_squared_error",
        cv=cv, random_state=SEED, n_jobs=-1, refit=False, verbose=1,
    )
    with timed("ExtraTrees RandomizedSearchCV"):
        search.fit(Xsearch, ysearch)
    best_params = {k: (v.item() if hasattr(v, "item") else v)
                   for k, v in search.best_params_.items()}
    print("best params:", best_params, "cv RMSE:", -search.best_score_)

    Xfit = pd.concat([Xtr_e, Xval_e], axis=0)
    yfit = np.concatenate([ytr, yval])
    final = ExtraTreesRegressor(**best_params, random_state=SEED, n_jobs=-1)
    with timed("ExtraTrees final fit (train+val)"):
        final.fit(Xfit, yfit)

    joblib.dump({"model": final, "encoder": encoder, "features": FEATURES},
                MODELS / "extra_trees_m5.joblib")
    pred = np.clip(final.predict(Xte_e), 0, None)
    metrics = regression_metrics(yte, pred)
    metrics["bootstrap"] = bootstrap_regression(yte, pred)
    metrics["baseline"] = {
        "seasonal_naive_lag7": regression_metrics(yte, mte["lag_7"].to_numpy()),
        "mean_train": regression_metrics(yte, np.full_like(yte, ytr.mean())),
    }
    metrics["per_store"] = {
        store: regression_metrics(sub[TARGET], np.clip(final.predict(Xte_e.loc[sub.index]), 0, None))
        for store, sub in mte.groupby("store_id", observed=True)
    }
    print("test:", {k: round(v, 4) for k, v in metrics.items()
                    if isinstance(v, float)})

    # staged validation curve (training replay)
    rng = np.random.default_rng(SEED + 1)
    vidx = rng.choice(len(Xval_e), min(8000, len(Xval_e)), replace=False)
    Xv = Xval_e.iloc[vidx]
    Xv_np = Xv.to_numpy()
    yv = yval[vidx]
    cum = np.zeros(len(Xv))
    hist = {"trees": [], "val_rmse": [], "val_mae": []}
    preds = []
    for i, est in enumerate(final.estimators_):
        preds.append(est.predict(Xv_np))
        if (i + 1) % 10 == 0 or i == len(final.estimators_) - 1:
            cum = np.mean(preds, axis=0)
            hist["trees"].append(i + 1)
            hist["val_rmse"].append(float(np.sqrt(np.mean((yv - cum) ** 2))))
            hist["val_mae"].append(float(np.mean(np.abs(yv - cum))))

    # feature importance
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": final.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(TABLES / "m5_feature_importance.csv", index=False)

    Xp = Xte_e.iloc[:2500].to_numpy()
    pi = permutation_importance(
        final, Xp, yte[:2500], n_repeats=2,
        scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1,
    )
    perm = pd.DataFrame({
        "feature": FEATURES, "perm_importance": pi.importances_mean,
        "perm_std": pi.importances_std,
    }).sort_values("perm_importance", ascending=False).reset_index(drop=True)
    perm.to_csv(TABLES / "m5_permutation_importance.csv", index=False)

    # 2D partial dependence of the ensemble on lag_7 x roll_mean_28
    med = Xtr_e.median()
    q = Xtr_e.quantile([0.05, 0.95])
    gx = np.linspace(q.loc[0.05, "lag_7"], q.loc[0.95, "lag_7"], 26)
    gy = np.linspace(q.loc[0.05, "roll_mean_28"], q.loc[0.95, "roll_mean_28"], 26)
    grid = pd.DataFrame([med.values] * (len(gx) * len(gy)), columns=FEATURES)
    grid["lag_7"] = np.repeat(gx, len(gy))
    grid["roll_mean_28"] = np.tile(gy, len(gx))
    z = np.clip(final.predict(grid), 0, None).reshape(len(gx), len(gy))

    # test sample for the web app + tree dispersion
    rng = np.random.default_rng(SEED + 2)
    sidx = rng.choice(len(Xte_e), min(400, len(Xte_e)), replace=False)
    Xs = Xte_e.iloc[sidx]
    test_sample = []
    tree_preds = np.array([est.predict(Xs.to_numpy()) for est in final.estimators_[:80]])
    for j, i in enumerate(sidx):
        row = mte.iloc[i]
        test_sample.append({
            "date": str(row["date"].date()),
            "store": str(row["store_id"]),
            "item": str(row["item_id"]),
            "actual": float(yte[i]),
            "pred": float(np.clip(pred[i], 0, None)),
            "uncertainty": float(tree_preds[:, j].std()),
            "lag_7": float(row["lag_7"]),
            "roll_mean_28": float(row["roll_mean_28"]),
            "price": float(row["sell_price"]),
        })

    # forecasts for three representative series
    top_ids = (mte.groupby("id", observed=True)[TARGET].mean()
               .sort_values(ascending=False).head(3).index.tolist())
    series = []
    for sid in top_ids:
        mask = (mte["id"] == sid).to_numpy()
        sub = mte[mask].sort_values("date")
        p = np.clip(final.predict(Xte_e.loc[sub.index]), 0, None)
        series.append({
            "id": sid,
            "store": str(sub["store_id"].iloc[0]),
            "item": str(sub["item_id"].iloc[0]),
            "dates": [str(d.date()) for d in sub["date"]],
            "actual": [float(v) for v in sub[TARGET]],
            "pred": [float(v) for v in p],
        })

    # demo forest exported to JavaScript (same encoding, compact size)
    rng = np.random.default_rng(SEED + 3)
    n_demo = min(180_000, len(Xfit))
    didx = rng.choice(len(Xfit), n_demo, replace=False)
    demo = ExtraTreesRegressor(
        n_estimators=44, max_depth=11, min_samples_leaf=3,
        max_features=0.5, random_state=SEED, n_jobs=-1,
    )
    with timed("ExtraTrees demo forest"):
        demo.fit(Xfit.iloc[didx], yfit[didx])
    demo_pred = np.clip(demo.predict(Xte_e), 0, None)
    demo_metrics = regression_metrics(yte, demo_pred)
    print("demo forest:", {k: round(v, 4) for k, v in demo_metrics.items() if isinstance(v, float)})

    # ---- figures -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    hb = ax.hexbin(yte, pred, gridsize=55, cmap="Blues", mincnt=1, linewidths=0.2)
    lim = [0, max(float(np.percentile(yte, 99.5)), float(np.percentile(pred, 99.5)))]
    ax.plot(lim, lim, color=ROSE, lw=1.4, ls="--", label="y = ŷ")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Ventas reales (unidades/día)")
    ax.set_ylabel("Ventas predichas (unidades/día)")
    ax.set_title("Extra Trees · M5: valores reales vs. predichos (test)")
    ax.text(0.04, 0.95, f"RMSE = {metrics['rmse']:.3f}\nR² = {metrics['r2']:.3f}",
            transform=ax.transAxes, va="top", fontsize=11, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cbd5e1"))
    fig.colorbar(hb, ax=ax, label="frecuencia")
    ax.legend(loc="lower right")
    savefig(fig, FIGURES / "m5_actual_vs_pred.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(hist["trees"], hist["val_rmse"], color=BLUE, lw=2.1, label="RMSE validación")
    ax.plot(hist["trees"], hist["val_mae"], color=TEAL, lw=1.8, label="MAE validación")
    ax.set_xlabel("Número de árboles (media acumulada)")
    ax.set_ylabel("Error (unidades/día)")
    ax.set_title("Curva de aprendizaje del ensamble Extra Trees")
    ax.legend()
    savefig(fig, FIGURES / "m5_learning_curve.png")

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    top = imp.head(16).iloc[::-1]
    colors = [BLUE if f in ("lag_1", "lag_7", "lag_14", "lag_28", "lag_364",
                            "roll_mean_7", "roll_mean_28", "roll_mean_90") else TEAL
              for f in top["feature"]]
    ax.barh(top["feature"], top["importance"], color=colors, height=0.68)
    ax.set_xlabel("Importancia por reducción de impureza (normalizada)")
    ax.set_title("Extra Trees · importancia de características")
    savefig(fig, FIGURES / "m5_importance.png")

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    levels = np.linspace(z.min(), z.max(), 18)
    cs = ax.contourf(gx, gy, z.T, levels=levels, cmap="viridis")
    ax.contour(gx, gy, z.T, levels=8, colors="white", linewidths=0.5, alpha=0.55)
    ax.set_xlabel("lag_7 (ventas de hace 7 días)")
    ax.set_ylabel("media móvil 28 días")
    ax.set_title("Superficie de respuesta (PDP 2D) del ensamble")
    fig.colorbar(cs, ax=ax, label="ventas predichas")
    savefig(fig, FIGURES / "m5_pdp_2d.png")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    Xg, Yg = np.meshgrid(gx, gy)
    surf = ax.plot_surface(Xg, Yg, z.T, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax.set_xlabel("lag_7"); ax.set_ylabel("media móvil 28d"); ax.set_zlabel("ventas predichas")
    ax.set_title("PDP 3D: interacción lag_7 × media móvil")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    savefig(fig, FIGURES / "m5_pdp_3d.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=False)
    for ax, s in zip(axes, series):
        ax.plot(range(1, 29), s["actual"], color=INK, marker="o", ms=3, lw=1.4, label="real")
        ax.plot(range(1, 29), s["pred"], color=AMBER, lw=2.0, label="Extra Trees")
        ax.set_title(f"{s['item']} · {s['store']}", fontsize=9.5)
        ax.set_xlabel("día de test")
        ax.grid(alpha=0.5)
    axes[0].set_ylabel("unidades/día")
    axes[0].legend(fontsize=8)
    fig.suptitle("Pronóstico de las 3 series de mayor volumen (28 días)", fontsize=12, y=1.02)
    savefig(fig, FIGURES / "m5_series_forecast.png")

    search_df = pd.DataFrame(search.cv_results_)
    search_df["rmse"] = -search_df["mean_test_score"]
    search_df = search_df.sort_values("rmse").reset_index(drop=True)
    search_df.insert(0, "rank", np.arange(1, len(search_df) + 1))
    search_df["config"] = search_df["params"].apply(
        lambda p: " | ".join(f"{k.replace('model__','')}={v}" for k, v in sorted(p.items())))
    search_df[["rank", "rmse", "std_test_score", "config"]].to_csv(
        TABLES / "m5_search_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    topn = search_df.head(8).iloc[::-1]
    ax.barh([f"#{i+1}" for i in range(len(topn))], topn["rmse"], color=BLUE, height=0.66)
    ax.set_xlabel("RMSE de validación (menor es mejor)")
    ax.set_title("RandomizedSearchCV · 12 configuraciones evaluadas")
    for i, v in enumerate(topn["rmse"]):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8.5, color=MUTED)
    savefig(fig, FIGURES / "m5_search.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(pred - yte, bins=60, color=BLUE, alpha=0.82)
    ax.axvline(0, color=ROSE, lw=1.3, ls="--")
    ax.set_xlabel("residuo (predicho − real)")
    ax.set_ylabel("frecuencia")
    ax.set_title(f"Distribución de residuos · sesgo medio = {metrics['bias']:.3f}")
    ax.set_yscale("log")
    savefig(fig, FIGURES / "m5_residuals.png")

    # ---- artifacts -----------------------------------------------------
    joblib.dump({"model": demo, "encoder": encoder, "features": FEATURES},
                MODELS / "extra_trees_m5_demo.joblib")

    medians = Xtr.median(numeric_only=True).to_dict()
    quantiles = {c: {"p05": float(Xtr[c].quantile(0.05)), "p50": float(Xtr[c].median()),
                     "p95": float(Xtr[c].quantile(0.95))}
                 for c in FEATURES if c not in CATEGORICAL}
    cat_levels = {c: [str(v) for v in encoder.categories_[FEATURES.index(c)]] for c in CATEGORICAL}

    result = {
        "model": "Extra Trees",
        "task": "regression",
        "dataset": "Walmart M5 (muestra estratificada)",
        "summary": summary,
        "metrics": metrics,
        "demo_metrics": demo_metrics,
        "best_params": best_params,
        "cv_rmse": float(-search.best_score_),
        "search": search_df[["rank", "rmse", "std_test_score", "config"]].head(12).to_dict("records"),
        "learning": hist,
        "importance": imp.head(20).to_dict("records"),
        "permutation": perm.head(20).to_dict("records"),
        "pdp": {"x": gx.tolist(), "y": gy.tolist(), "z": z.tolist(),
                "x_label": "lag_7", "y_label": "roll_mean_28"},
        "test_sample": test_sample,
        "series": series,
        "medians": medians,
        "quantiles": quantiles,
        "cat_levels": cat_levels,
        "feature_order": FEATURES,
        "categorical": CATEGORICAL,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(result, TABLES / "extra_trees_results.json")
    print(f"Extra Trees finished in {result['runtime_seconds']}s")
    return result


if __name__ == "__main__":
    run()
