"""Walmart M5 data preparation for daily unit-sales regression.

The sales extract used here contains 30,490 item-store series x 1,913 days
(2011-01-29 to 2016-04-24); the calendar file spans 1,969 days through 2016-06-19.
For a reproducible classroom experiment we keep a stratified sample of stores
(80 series per store) and the last 500 days, while every lag/rolling feature is
computed with the full preceding history.  Target-encoding style features are
estimated only on the period *before* the modelling window to avoid leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    M5_CALENDAR,
    M5_PRICES,
    M5_SALES,
    M5_SERIES_PER_STORE,
    M5_TEST_DAYS,
    M5_VAL_DAYS,
    M5_WINDOW_DAYS,
    SEED,
)

CATEGORICAL = ["dept_id", "cat_id", "store_id", "state_id"]
NUMERIC = [
    "wday", "month", "dayofyear", "is_weekend", "snap", "event_flag",
    "event_type_code", "sell_price", "price_change_28", "price_ratio_item",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_364",
    "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
    "roll_mean_90", "roll_max_28", "zero_ratio_28",
    "item_mean_sales", "item_store_mean_sales", "dept_mean_sales",
    "store_mean_sales", "item_active_days", "item_price_mean",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "sales"


def _read_sales_sample() -> tuple[pd.DataFrame, list[str]]:
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    head = pd.read_csv(M5_SALES, nrows=0)
    all_cols = list(head.columns)
    d_cols = [c for c in all_cols if c.startswith("d_")]

    sample = pd.read_csv(M5_SALES, usecols=meta_cols)
    # choose active series using the last 500 days
    tail = pd.read_csv(M5_SALES, usecols=["id"] + d_cols[-M5_WINDOW_DAYS:])
    activity = tail[d_cols[-M5_WINDOW_DAYS:]].sum(axis=1)
    sample["_activity"] = activity

    chosen = []
    for i, (store, grp) in enumerate(sample.groupby("store_id", observed=True)):
        grp = grp[grp["_activity"] > 0]
        n = min(M5_SERIES_PER_STORE, len(grp))
        chosen.append(grp.sample(n=n, random_state=SEED + i))
    chosen_ids = set(pd.concat(chosen)["id"])

    dtype = {c: "int16" for c in d_cols}
    full = pd.read_csv(M5_SALES, dtype=dtype)
    full = full[full["id"].isin(chosen_ids)].copy()
    return full, d_cols


def build_m5_dataset() -> dict:
    sales, d_cols = _read_sales_sample()
    calendar = pd.read_csv(M5_CALENDAR)
    prices = pd.read_csv(M5_PRICES)

    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["dayofyear"] = calendar["date"].dt.dayofyear
    calendar["is_weekend"] = calendar["wday"].isin([1, 2]).astype(np.int8)

    # modelling window and history boundaries
    n_days = len(d_cols)
    window_start_idx = n_days - M5_WINDOW_DAYS           # first modelled day
    feature_start_idx = window_start_idx - 365           # need lag_364

    hist_cols = d_cols[:window_start_idx]
    feature_cols = d_cols[feature_start_idx:]

    hist = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + hist_cols]
    hist_values = hist[hist_cols].to_numpy(dtype=np.float32)
    hist = hist[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    hist["item_mean_sales"] = hist_values.mean(axis=1)
    hist["item_active_days"] = (hist_values > 0).sum(axis=1)

    item_store = hist[["id", "item_mean_sales"]].rename(columns={"item_mean_sales": "item_store_mean_sales"})
    item_stats = (hist.groupby("item_id", observed=True)["item_mean_sales"]
                  .mean().rename("item_mean_sales").reset_index())
    dept_stats = (hist.groupby("dept_id", observed=True)["item_mean_sales"]
                  .mean().rename("dept_mean_sales").reset_index())
    store_stats = (hist.groupby("store_id", observed=True)["item_mean_sales"]
                   .mean().rename("store_mean_sales").reset_index())
    active_stats = (hist.groupby("item_id", observed=True)["item_active_days"]
                    .mean().rename("item_active_days").reset_index())

    # historical item price level (only before the modelled window)
    week_cut = int(calendar.loc[window_start_idx, "wm_yr_wk"])
    price_hist = prices[prices["wm_yr_wk"] < week_cut]
    item_price = (price_hist.groupby("item_id", observed=True)["sell_price"]
                  .median().rename("item_price_mean").reset_index())

    long = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + feature_cols].melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d", value_name=TARGET,
    )
    long = long.merge(calendar[["d", "date", "wm_yr_wk", "wday", "month", "dayofyear",
                                "is_weekend", "event_name_1", "event_type_1",
                                "event_name_2", "event_type_2", "snap_CA", "snap_TX",
                                "snap_WI"]], on="d", how="left")
    long["state_id"] = long["state_id"].astype(str)

    long["snap"] = np.select(
        [long["state_id"].eq("CA"), long["state_id"].eq("TX"), long["state_id"].eq("WI")],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]], default=0,
    ).astype(np.int8)
    long["event_flag"] = (long["event_name_1"].notna() | long["event_name_2"].notna()).astype(np.int8)
    event_map = {"Cultural": 1, "National": 2, "Religious": 3, "Sporting": 4}
    ev = long["event_type_1"].fillna(long["event_type_2"])
    long["event_type_code"] = ev.map(event_map).fillna(0).astype(np.int8)

    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    g = long.groupby("id", observed=True, sort=False)[TARGET]
    long["lag_1"] = g.shift(1)
    long["lag_7"] = g.shift(7)
    long["lag_14"] = g.shift(14)
    long["lag_28"] = g.shift(28)
    long["lag_364"] = g.shift(364)
    for w in (7, 28, 90):
        long[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
    long["roll_std_7"] = g.shift(1).rolling(7).std().reset_index(level=0, drop=True)
    long["roll_std_28"] = g.shift(1).rolling(28).std().reset_index(level=0, drop=True)
    long["roll_max_28"] = g.shift(1).rolling(28).max().reset_index(level=0, drop=True)
    long["zero_ratio_28"] = (g.shift(1).rolling(28).apply(lambda x: np.mean(x == 0), raw=True)
                             .reset_index(level=0, drop=True))

    # weekly price observed on the same day (known in advance in M5)
    pairs = long[["store_id", "item_id"]].drop_duplicates()
    price_sub = prices.merge(pairs, on=["store_id", "item_id"], how="inner")
    long = long.merge(price_sub, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    long["sell_price"] = (long.groupby("id", observed=True)["sell_price"]
                          .transform(lambda s: s.ffill().bfill()))
    long["sell_price"] = long["sell_price"].fillna(long.groupby("item_id", observed=True)["sell_price"]
                                                   .transform("median"))
    long["sell_price"] = long["sell_price"].fillna(long["sell_price"].median())
    long["price_change_28"] = (long["sell_price"] -
                               long.groupby("id", observed=True)["sell_price"].shift(28))

    long = long.merge(item_store, on="id", how="left")
    long = long.merge(item_stats, on="item_id", how="left")
    long = long.merge(dept_stats, on="dept_id", how="left")
    long = long.merge(store_stats, on="store_id", how="left")
    long = long.merge(active_stats, on="item_id", how="left")
    long = long.merge(item_price, on="item_id", how="left")
    long["price_ratio_item"] = long["sell_price"] / long["item_price_mean"].replace(0, np.nan)

    window_start_date = calendar.loc[window_start_idx, "date"]
    df = long[long["date"] >= window_start_date].copy()
    df = df.dropna(subset=["lag_364", "roll_mean_90"]).reset_index(drop=True)

    # time-ordered split
    dates = np.sort(df["date"].unique())
    test_dates = dates[-M5_TEST_DAYS:]
    val_dates = dates[-(M5_TEST_DAYS + M5_VAL_DAYS):-M5_TEST_DAYS]
    train_dates = dates[:-(M5_TEST_DAYS + M5_VAL_DAYS)]

    df["split"] = np.where(df["date"].isin(test_dates), "test",
                           np.where(df["date"].isin(val_dates), "val", "train"))

    for c in CATEGORICAL:
        df[c] = df[c].astype(str).astype("category")

    df = df.sort_values(["date", "id"]).reset_index(drop=True)
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                 "date", "d", "split", TARGET, "lag_7", "roll_mean_28", "sell_price"]
    summary = {
        "n_series": int(df["id"].nunique()),
        "n_stores": int(df["store_id"].nunique()),
        "n_items": int(df["item_id"].nunique()),
        "n_days": int(df["date"].nunique()),
        "n_rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "train_rows": int((df["split"] == "train").sum()),
        "val_rows": int((df["split"] == "val").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "target_mean": float(df[TARGET].mean()),
        "target_std": float(df[TARGET].std()),
        "target_zero_share": float((df[TARGET] == 0).mean()),
        "target_max": float(df[TARGET].max()),
        "features": FEATURES,
    }
    return {"frame": df, "meta_cols": meta_cols, "features": FEATURES,
            "summary": summary, "train_dates": train_dates, "val_dates": val_dates,
            "test_dates": test_dates}


def split_xy(df: pd.DataFrame, features: list[str]):
    parts = {}
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        parts[name] = (sub[features].copy(), sub[TARGET].to_numpy(dtype=float), sub)
    return parts
