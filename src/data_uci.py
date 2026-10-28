"""UCI 'Predict students' dropout and academic success' dataset."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import SEED, UCI_CSV, UCI_TEST_SIZE

TARGET = "Target"
TARGET_ORDER = ["Dropout", "Enrolled", "Graduate"]

# Codebook variables that represent nominal categories (not quantities).
CATEGORICAL = [
    "Marital_status", "Application_mode", "Application_order", "Course",
    "Daytime_evening_attendance", "Previous_qualification", "Nacionality",
    "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
    "Fathers_occupation", "Displaced", "Educational_special_needs", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "International",
]

NUMERIC = [
    "Previous_qualification_grade", "Admission_grade", "Age_at_enrollment",
    "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate", "Inflation_rate", "GDP",
]


def _clean(name: str) -> str:
    name = name.replace("\t", " ").replace("'", "").strip()
    name = re.sub(r"[()]", "", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def build_uci_dataset() -> dict:
    df = pd.read_csv(UCI_CSV, sep=";", encoding="utf-8-sig")
    df.columns = [_clean(c) for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    # A few derived ratios used in the literature on academic success.
    df["Approved_ratio_1st"] = (df["Curricular_units_1st_sem_approved"] /
                                df["Curricular_units_1st_sem_enrolled"].replace(0, np.nan)).fillna(0)
    df["Approved_ratio_2nd"] = (df["Curricular_units_2nd_sem_approved"] /
                                df["Curricular_units_2nd_sem_enrolled"].replace(0, np.nan)).fillna(0)
    df["Grade_delta"] = df["Curricular_units_2nd_sem_grade"] - df["Curricular_units_1st_sem_grade"]
    derived = ["Approved_ratio_1st", "Approved_ratio_2nd", "Grade_delta"]

    features = CATEGORICAL + NUMERIC + derived
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)

    summary = {
        "n_rows": int(len(df)),
        "n_features": len(features),
        "n_categorical": len(CATEGORICAL),
        "n_numeric": len(NUMERIC) + len(derived),
        "target_counts": {k: int(v) for k, v in df[TARGET].value_counts().items()},
        "target_order": TARGET_ORDER,
        "features": features,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC + derived,
        "target": TARGET,
        "test_size": UCI_TEST_SIZE,
        "seed": SEED,
    }
    return {"frame": df, "features": features, "categorical": CATEGORICAL,
            "numeric": NUMERIC + derived, "target": TARGET, "summary": summary}
