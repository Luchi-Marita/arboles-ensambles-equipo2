"""Download and prepare the three public datasets used by the project.

UCI and CDC are downloaded from stable public URLs. The M5 competition files
require a Kaggle account: the script uses the Kaggle CLI if it is available.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

M5_FILES = {
    "sales_train_validation.csv": RAW / "m5_sales.csv",
    "calendar.csv": RAW / "m5_calendar.csv",
    "sell_prices.csv": RAW / "m5_prices.csv",
}
CDC_CSV = RAW / "cdc_diabetes.csv"
UCI_CSV = RAW / "uci_dropout" / "data.csv"
UCI_ZIP = RAW / "uci_dropout.zip"

CDC_URL = (
    "https://raw.githubusercontent.com/BabsMatt/Diabetes-Health-Indicators/"
    "master/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"descargando {url}")
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _check() -> list[str]:
    missing = []
    for path in (*M5_FILES.values(), CDC_CSV, UCI_CSV):
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    return missing


def _download_cdc() -> None:
    if CDC_CSV.exists() and CDC_CSV.stat().st_size > 1_000_000:
        print(f"CDC listo: {CDC_CSV}")
        return
    _download(CDC_URL, CDC_CSV)
    header = CDC_CSV.open(encoding="utf-8").readline()
    if "Diabetes_binary" not in header:
        raise SystemExit(f"El CSV de CDC no tiene el encabezado esperado: {header[:120]}")


def _download_uci() -> None:
    if UCI_CSV.exists() and UCI_CSV.stat().st_size > 100_000:
        print(f"UCI listo: {UCI_CSV}")
        return
    if not UCI_ZIP.exists():
        _download(UCI_URL, UCI_ZIP)
    tmp = RAW / "_uci_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(UCI_ZIP) as zf:
        zf.extractall(tmp)
    matches = list(tmp.rglob("data.csv"))
    if not matches:
        raise SystemExit("No se encontró data.csv dentro del ZIP de UCI.")
    UCI_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(matches[0], UCI_CSV)
    shutil.rmtree(tmp)
    print(f"  -> {UCI_CSV}")


def _download_m5() -> None:
    if all(path.exists() for path in M5_FILES.values()):
        print("M5 listo: sales, calendar y prices.")
        return

    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise SystemExit(
            "Faltan los archivos M5.\n"
            "Instala la CLI de Kaggle (`pip install kaggle`), configura "
            "KAGGLE_USERNAME/KAGGLE_KEY y vuelve a ejecutar este script.\n"
            "Alternativa: descarga manualmente `sales_train_validation.csv`, "
            "`calendar.csv` y `sell_prices.csv` desde la competencia "
            "`m5-forecasting-accuracy` y guárdalos en data/raw con los nombres "
            "m5_sales.csv, m5_calendar.csv y m5_prices.csv."
        )

    zips = sorted(RAW.glob("m5*.zip"))
    if not zips:
        subprocess.run(
            [kaggle, "competitions", "download", "-c", "m5-forecasting-accuracy",
             "-p", str(RAW), "--force"],
            check=True,
        )
        zips = sorted(RAW.glob("m5*.zip"))
    if not zips:
        raise SystemExit("Kaggle no devolvió ningún ZIP para M5.")

    tmp = RAW / "_m5_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for archive in zips:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)

    for source_name, dest in M5_FILES.items():
        matches = list(tmp.rglob(source_name))
        if not matches:
            raise SystemExit(f"No se encontró {source_name} dentro de los ZIP de M5.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matches[0], dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(tmp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga los datasets del proyecto.")
    parser.add_argument("--check", action="store_true",
                        help="Solo verifica los archivos, sin descargar.")
    args = parser.parse_args()

    missing = _check()
    if args.check:
        if missing:
            print("Faltan:\n" + "\n".join(f"  - {m}" for m in missing))
            raise SystemExit(1)
        print("Todos los datasets están presentes.")
        return

    _download_cdc()
    _download_uci()
    _download_m5()

    missing = _check()
    if missing:
        raise SystemExit("Aún faltan archivos:\n" + "\n".join(missing))
    print("Datasets listos en data/raw/.")


if __name__ == "__main__":
    main()
