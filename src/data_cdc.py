"""CDC BRFSS 2015 diabetes health indicators (balanced 50/50 sample)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CDC_CSV, CDC_TEST_SIZE, SEED

TARGET = "Diabetes_binary"
BASE_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
ENGINEERED = ["Comorbidity", "HealthyHabits", "BMI_cat", "MentPhysDays"]
FEATURES = BASE_FEATURES + ENGINEERED


def build_cdc_dataset() -> dict:
    df = pd.read_csv(CDC_CSV)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    df["Comorbidity"] = (df["HighBP"] + df["HighChol"] + df["Stroke"] +
                         df["HeartDiseaseorAttack"]).astype(np.int8)
    df["HealthyHabits"] = (df["PhysActivity"] + df["Fruits"] + df["Veggies"] -
                           df["Smoker"] - df["HvyAlcoholConsump"]).astype(np.int8)
    df["BMI_cat"] = pd.cut(df["BMI"], bins=[-np.inf, 18.5, 25, 30, 35, np.inf],
                           labels=False).astype(np.int8)
    df["MentPhysDays"] = (df["MentHlth"] + df["PhysHlth"]).astype(np.int16)

    summary = {
        "n_rows": int(len(df)),
        "n_features": len(BASE_FEATURES),
        "n_features_engineered": len(FEATURES),
        "target_rate": float(df[TARGET].mean()),
        "duplicates_removed": int(pd.read_csv(CDC_CSV).duplicated().sum()),
        "bmi_mean": float(df["BMI"].mean()),
        "age_mean": float(df["Age"].mean()),
        "features": FEATURES,
        "target": TARGET,
        "test_size": CDC_TEST_SIZE,
        "seed": SEED,
    }
    return {"frame": df, "features": FEATURES, "target": TARGET, "summary": summary}
