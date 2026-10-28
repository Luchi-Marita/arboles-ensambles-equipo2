# Bosque de decisiones · Árboles y Ensambles

Trabajo final de **Aprendizaje Supervisado · Familia 1: Árboles y Ensambles**.
El proyecto entrena y compara tres ensambles en tres problemas reales, exporta la inferencia a JavaScript y publica un dashboard estático con el informe técnico.

- **Dashboard:** https://luchi-marita.github.io/arboles-ensambles-equipo2/
- **Repositorio:** https://github.com/Luchi-Marita/arboles-ensambles-equipo2
- **Informe:** `docs/reports/Informe_Tecnico_Arboles_Ensambles.pdf` y `.docx`
- **Autor:** Luchi-Marita · **Equipo:** Equipo 2

## Resultados principales

| Modelo | Caso | Tarea | Métrica principal | Métricas de apoyo |
|---|---|---|---|---|
| Extra Trees | Walmart M5 | Regresión de ventas diarias | RMSE **1.998** | MAE 0.990 · R² 0.607 · WAPE 70.2% |
| AdaBoost | CDC BRFSS 2015 | Clasificación binaria | ROC-AUC **0.821** | F1 0.758 @0.50 · F1 0.773 @0.38 |
| CatBoost | UCI Student Dropout | Clasificación multiclase | Macro-F1 **0.693** | Accuracy 0.731 · AUC-OvR 0.886 |

Extra Trees reduce el RMSE un **28%** frente al baseline estacional `lag_7` (2.776 → 1.998). AdaBoost prioriza recall con umbral 0.38 (recall 0.912, precisión 0.670). CatBoost detecta bien Dropout y Graduate; la clase intermedia Enrolled es la más difícil.

## Contenido del repositorio

```text
.
├── docs/                     # dashboard estático para GitHub Pages
│   ├── index.html
│   ├── assets/{css,js}
│   ├── data/dashboard.json
│   ├── models/*.json         # modelos de inferencia en el navegador
│   ├── img/
│   └── reports/              # informe PDF y DOCX publicados
├── src/
│   ├── config.py
│   ├── data_m5.py
│   ├── data_cdc.py
│   ├── data_uci.py
│   ├── train_extra_trees.py
│   ├── train_adaboost.py
│   ├── train_catboost.py
│   ├── recompute_extra_learning.py
│   ├── export_web.py
│   └── make_report.py
├── scripts/
│   ├── download_data.py
│   └── convert_pdf.ps1
├── artifacts/
│   ├── figures/
│   ├── tables/
│   └── models/               # joblib locales, ignorados por Git
├── reports/                  # DOCX/PDF y recursos generados
├── run_all.py
└── requirements.txt
```

## Inicio rápido

Requisitos: Python 3.12+ (probado con 3.14.6), Microsoft Word en Windows para generar el PDF, y una cuenta de Kaggle para M5.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
python run_all.py
```

`run_all.py` ejecuta en orden:

1. búsqueda y entrenamiento de Extra Trees;
2. recálculo de la curva de aprendizaje sobre el test completo;
3. búsqueda y entrenamiento de AdaBoost;
4. búsqueda y entrenamiento de CatBoost;
5. exportación de modelos y dashboard a `docs/`.

El pipeline completo tarda aproximadamente 25–30 minutos en un equipo personal y necesita ~3 GB libres para el joblib de Extra Trees.

### Datos

| Fuente | Archivo esperado | Origen |
|---|---|---|
| Walmart M5 | `data/raw/m5_sales.csv`, `m5_calendar.csv`, `m5_prices.csv` | [Kaggle · M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) |
| CDC BRFSS 2015 | `data/raw/cdc_diabetes.csv` | [Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| UCI Student Dropout | `data/raw/uci_dropout/data.csv` | [UCI Dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |

`scripts/download_data.py` descarga CDC y UCI automáticamente. Para M5 usa la CLI de Kaggle (`kaggle competitions download -c m5-forecasting-accuracy`); si no está configurada, descarga manualmente los tres CSV y renómbralos como se indica. La extracción de ventas usada contiene 1.913 días (2011-01-29 a 2016-04-24); el calendario cubre 1.969 días.

### Dashboard local

```powershell
python -m http.server 8765 --directory docs
```

Abrir `http://localhost:8765/`. El sitio es completamente estático: no requiere backend ni conexión a un servidor de modelos.

### Informe

```powershell
python -m src.make_report
powershell -ExecutionPolicy Bypass -File scripts/convert_pdf.ps1
```

El primer comando genera el DOCX y las ecuaciones; el segundo actualiza el índice con Word y exporta el PDF. Los archivos finales se copian a `docs/reports/` para que los enlaces del dashboard funcionen.

## Metodología y reproducibilidad

- Semilla fija `42` y rutas centralizadas en `src/config.py`.
- M5: muestra estratificada de 800 series (80 por tienda), ventana de 500 días, 32 variables y split temporal 355.200 / 22.400 / 22.400. El modelo final se reentrena con train+validación y se evalúa una sola vez en test.
- CDC: 69.057 filas tras deduplicar, 25 variables y split estratificado 80/20. El umbral 0.38 se elige en una partición interna del entrenamiento, nunca en test.
- UCI: 4.424 estudiantes, 39 variables (18 categóricas nativas para CatBoost) y split estratificado 80/20.
- Intervalos bootstrap para RMSE, R², AUC y F1.
- Paridad de inferencia Python/JavaScript verificada: diferencia máxima < `6e-17`.
- El dashboard usa un bosque Extra Trees demo de 44 árboles y un CatBoost demo one-hot; las métricas del informe corresponden a los modelos completos. Esta diferencia está indicada en la interfaz.

## Límites

- La muestra de M5 no cubre las 30.490 series; el modelo suaviza picos y no extrapola tendencias.
- El dataset CDC es autodeclarado y está balanceado 50/50; las probabilidades no son clínicamente calibradas.
- UCI es un snapshot con 885 estudiantes en test; la clase Enrolled es ambigua.
- Ningún modelo está autorizado para decisiones clínicas, financieras o académicas automáticas. Requieren validación externa, revisión humana y auditoría de equidad.

## Atribución

Los datos pertenecen a Walmart/Kaggle, CDC/BRFSS y UCI Machine Learning Repository. El código y el informe se comparten con fines académicos.
