"""Recompute the Extra Trees learning curve on the held-out test split.

The original curve was computed on the validation split after that split had
been used in the final fit, so it measured training error. This script keeps
the final model untouched and replays its trees over a test sample.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS, SEED, TABLES
from src.data_m5 import FEATURES, build_m5_dataset, split_xy
from src.utils import load_json, save_json, timed

SAMPLE = 22_400


def main() -> None:
    results_path = TABLES / "extra_trees_results.json"
    results = load_json(results_path)
    bundle = joblib.load(MODELS / "extra_trees_m5.joblib")
    model, encoder = bundle["model"], bundle["encoder"]

    data = build_m5_dataset()
    parts = split_xy(data["frame"], FEATURES)
    _, yte, _ = parts["test"]
    Xte = parts["test"][0]
    Xte_e = pd.DataFrame(encoder.transform(Xte), columns=FEATURES, index=Xte.index)
    rng = np.random.default_rng(SEED + 1)
    idx = rng.choice(len(Xte_e), min(SAMPLE, len(Xte_e)), replace=False)
    Xv = Xte_e.iloc[idx].to_numpy()
    yv = yte[idx]

    hist = {"trees": [], "val_rmse": [], "val_mae": []}
    preds = []
    with timed("replay Extra Trees on held-out test"):
        for i, est in enumerate(model.estimators_):
            preds.append(est.predict(Xv))
            if (i + 1) % 10 == 0 or i == len(model.estimators_) - 1:
                cum = np.mean(preds, axis=0)
                hist["trees"].append(i + 1)
                hist["val_rmse"].append(float(np.sqrt(np.mean((yv - cum) ** 2))))
                hist["val_mae"].append(float(np.mean(np.abs(yv - cum))))

    results["learning"] = hist
    results["learning_split"] = "test (held-out, full)"
    save_json(results, results_path)

    import matplotlib.pyplot as plt
    from .config import FIGURES
    from .plot_style import BLUE, TEAL, savefig, setup

    setup()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(hist["trees"], hist["val_rmse"], color=BLUE, lw=2.1, label="RMSE test")
    ax.plot(hist["trees"], hist["val_mae"], color=TEAL, lw=1.8, label="MAE test")
    ax.set_xlabel("Número de árboles (media acumulada)")
    ax.set_ylabel("Error (unidades/día)")
    ax.set_title("Curva de aprendizaje del ensamble Extra Trees (test)")
    ax.legend()
    savefig(fig, FIGURES / "m5_learning_curve.png")
    plt.close(fig)

    print("first/last RMSE:", round(hist["val_rmse"][0], 4), round(hist["val_rmse"][-1], 4))
    print("first/last MAE:", round(hist["val_mae"][0], 4), round(hist["val_mae"][-1], 4))


if __name__ == "__main__":
    main()
