# Fase 1 — Carga y split train/test

## Objetivo de la fase
Cargar `data/raw/housing.csv` quedándonos solo con `Latitude`, `Longitude` y `MedInc`, y dividir en train (80%) y test (20%) de forma reproducible.

## Utilidad en el flujo
- Reduce el problema a posición + ingreso medio, que es lo que pide el enunciado para regionalizar casas.
- El split permite entrenar el no-supervisado solo con train y reservar test para validar que los clusters generalizan a puntos nuevos (Fase 3) y para evaluar el clasificador supervisado (Fase 4) sin fuga de información.
- Resultado real: 20.640 filas → train 16.512, test 4.128.

## Tecnología y por qué
- `pandas.read_csv`: estándar tabular, carga directa del CSV descargado desde `https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv`. Alternativa `numpy.loadtxt` descartada por no manejar cabeceras ni columnas por nombre.
- `sklearn.model_selection.train_test_split(test_size=0.2, random_state=42)`: split aleatorio reproducible con una línea. No se estratifica porque aún no hay etiquetas. `random_state=42` garantiza que train/test sean comparables entre ejecuciones.
- Descarga con fallback en `src/data_loader.py::load_housing`: si `data/raw/housing.csv` no existe, lo descarga de la URL y lo cachea. Así el repo funciona clonado desde cero.

## Salidas
- `src/data_loader.py`
- `data/raw/housing.csv`, `data/processed/train_labeled.csv` y `test_labeled.csv` (estos últimos se escriben en Fases 2-3).

## Cómo ejecutar
```bash
python3 src/app.py
# o solo esta fase:
python3 -c "from src.data_loader import load_housing, split_data; df=load_housing('data/raw/housing.csv'); print(split_data(df)[0].shape)"
```
