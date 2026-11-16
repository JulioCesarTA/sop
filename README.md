# Pronóstico de series de tiempo con Prophet

Cuatro clases prácticas sobre la librería **Prophet** (de Facebook/Meta), con datos reales y actualizados de la Reserva Federal de EE. UU. (FRED). Cada clase tiene su notebook de Jupyter y una guía en Markdown que explica el código, los resultados y los gráficos paso a paso.

| Clase | Tema | Datos | Notebook | Guía |
|---|---|---|---|---|
| 1 | Prophet básico | Ventas de cerveza, vino y licor (1992–2026) | `01_Prophet_Basico_BeerWineLiquor.ipynb` | `01_Explicacion_Prophet_Basico.md`, `01b_Explicacion_3_Graficos.md` |
| 2 | Evaluación (train/test, RMSE, validación cruzada) | Millas recorridas en EE. UU. (1970–2026) | `02_Prophet_Evaluacion_MilesTraveled.ipynb` | `02_Explicacion_Prophet_Evaluacion.md` |
| 3 | Puntos de cambio de la tendencia | Empleo en ocio y hospitalidad, California (1990–2026) | `03_Prophet_Tendencia_Hospitality.ipynb` | `03_Explicacion_Prophet_Tendencia.md` |
| 4 | Estacionalidad aditiva vs. multiplicativa | Ventas por internet y catálogo (1992–2026) | `04_Prophet_Estacionalidad_VentasInternet.ipynb` | `04_Explicacion_Prophet_Estacionalidad.md`, `04b_Explicacion_Graficos_Estacionalidad.md` |

## Datos

Todos en la carpeta `Data/`, descargados de FRED:

| Archivo | Serie FRED | Qué mide |
|---|---|---|
| `BeerWineLiquor.csv` | MRTSSM4453USN | Ventas minoristas en tiendas de cerveza, vino y licor (millones de USD) |
| `Miles_Traveled.csv` | TRFVOLUSM227NFWA | Millas recorridas por vehículos en EE. UU. (millones) |
| `HospitalityEmployees.csv` | CALEIHN | Empleados de ocio y hospitalidad en California (miles) |
| `ventas_internet.csv` | MRTSSM4541USN | Ventas por internet y catálogo (millones de USD) |

## Cómo ejecutarlo

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Después abre cualquier notebook en VS Code o Jupyter y ejecuta todas las celdas.
