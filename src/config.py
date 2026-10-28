"""Central configuration for the project."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"
MODELS = ARTIFACTS / "models"
DOCS = ROOT / "docs"
DOCS_DATA = DOCS / "data"
DOCS_MODELS = DOCS / "models"
DOCS_IMG = DOCS / "img"
REPORTS = ROOT / "reports"

SEED = 42

M5_SALES = DATA_RAW / "m5_sales.csv"
M5_CALENDAR = DATA_RAW / "m5_calendar.csv"
M5_PRICES = DATA_RAW / "m5_prices.csv"
CDC_CSV = DATA_RAW / "cdc_diabetes.csv"
UCI_CSV = DATA_RAW / "uci_dropout" / "data.csv"

# M5 modelling window (the last 500 days of the competition history)
M5_SERIES_PER_STORE = 80
M5_WINDOW_DAYS = 500
M5_TEST_DAYS = 28
M5_VAL_DAYS = 28
M5_SEARCH_ROWS = 120_000

CDC_TEST_SIZE = 0.20
UCI_TEST_SIZE = 0.20

for _p in (DATA_PROC, ARTIFACTS, FIGURES, TABLES, MODELS, DOCS, DOCS_DATA,
           DOCS_MODELS, DOCS_IMG, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)
