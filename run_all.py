"""Run the full training and web-export pipeline.

Each step runs in a separate Python process so the large Extra Trees model is
released before the next experiment starts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = [
    ROOT / "data" / "raw" / "m5_sales.csv",
    ROOT / "data" / "raw" / "m5_calendar.csv",
    ROOT / "data" / "raw" / "m5_prices.csv",
    ROOT / "data" / "raw" / "cdc_diabetes.csv",
    ROOT / "data" / "raw" / "uci_dropout" / "data.csv",
]
STEPS = [
    ("Extra Trees: búsqueda y entrenamiento", "src.train_extra_trees"),
    ("Extra Trees: curva sobre test", "src.recompute_extra_learning"),
    ("AdaBoost: búsqueda y entrenamiento", "src.train_adaboost"),
    ("CatBoost: búsqueda y entrenamiento", "src.train_catboost"),
    ("Exportación del dashboard", "src.export_web"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline completo.")
    parser.add_argument("--check", action="store_true",
                        help="Verifica datos y muestra los pasos, sin entrenar.")
    args = parser.parse_args()

    missing = [str(path.relative_to(ROOT)) for path in DATA if not path.exists()]
    if missing:
        print("Faltan datos crudos:")
        for path in missing:
            print(f"  - {path}")
        print("Ejecuta: python scripts/download_data.py")
        return 1

    if args.check:
        print("Datos listos. Pasos:")
        for index, (label, module) in enumerate(STEPS, 1):
            print(f"  {index}. {label} ({module})")
        return 0

    started = time.perf_counter()
    for index, (label, module) in enumerate(STEPS, 1):
        print(f"\n[{index}/{len(STEPS)}] {label}")
        step_start = time.perf_counter()
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
        print(f"[ok] {label} ({time.perf_counter() - step_start:.1f} s)")

    print(f"\nPipeline completo en {time.perf_counter() - started:.1f} s.")
    print("Dashboard: python -m http.server 8765 --directory docs")
    print("Informe:   python -m src.make_report")
    print("PDF:       powershell -ExecutionPolicy Bypass -File scripts/convert_pdf.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
