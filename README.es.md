# Calificador de casas — California Housing

Clasificación de casas según su región e ingreso medio a partir del conjunto de datos California Housing (censo de California de 1990). Flujo en 5 fases: carga → K-Means (k=6) → predicción en test → clasificación supervisada → guardado de modelos.

## Dataset

- `data/raw/housing.csv` (20.640 filas), descargable desde `https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv`. Si no existe, `src/data_loader.py` lo descarga solo.
- Solo se usan las columnas `Latitude`, `Longitude` y `MedInc`.
- Split reproducible 80/20: train 16.512, test 4.128 (`train_test_split`, `random_state=42`).

## Pipeline por fases

1. **Carga y split** (`src/data_loader.py`): recorte a 3 columnas y división train/test.
2. **K-Means k=6** (`src/clustering.py`): `StandardScaler` ajustado solo en train + `KMeans(n_clusters=6, n_init=10)`; columna nueva `cluster` (int 0-5); scatter `Longitude` vs `Latitude` en `reports/figures/clusters_train.png`.
3. **Predicción en test** (`src/clustering.py::predict_clusters`): `transform` + `predict` sin reentrenar; overlay en `reports/figures/clusters_train_test.png`.
4. **Clasificación supervisada** (`src/classification.py`): `X=[Latitude, Longitude, MedInc]` escaladas, `y=cluster`. Comparativa en test:
   - Regresión logística: 0,9879
   - KNN (k=15): 0,9898
   - RandomForest (200 árboles): **0,9956** ← modelo final
5. **Guardado** (`joblib` en `models/`): `kmeans.pkl`, `scaler.pkl`, `classifier.pkl` + `metadata.json`.

Detalle de cada fase (utilidad + tecnología y por qué) en `docs/`: `FASE_01` a `FASE_05` y `docs/README.md`.

## Estructura

- **`src/app.py`** → orquestador de las 5 fases.
- **`src/data_loader.py`** → carga y split.
- **`src/clustering.py`** → K-Means + predicción.
- **`src/classification.py`** → comparativa de clasificadores.
- **`src/visualization.py`** → scatter train y train+test.
- **`src/explore.ipynb`** → notebook con las 5 fases para exploración.
- **`docs/`** → documentación por fase (qué hace, utilidad, tech y por qué).
- **`models/`** → `kmeans.pkl`, `scaler.pkl`, `classifier.pkl`, `metadata.json`.
- **`data/`** → `raw/housing.csv`, `processed/train_labeled.csv`, `processed/test_labeled.csv`.
- **`reports/`** → `metrics.md`, `figures/clusters_train.png`, `figures/clusters_train_test.png`.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python src/app.py
```

Reutilizar los modelos guardados:

```python
import joblib
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")
clf = joblib.load("models/classifier.pkl")
x = scaler.transform([[34.05, -118.25, 5.0]])  # Latitude, Longitude, MedInc
print(kmeans.predict(x), clf.predict(x))
```

## Notas técnicas

- El escalado con `StandardScaler` antes de K-Means es imprescindible: `MedInc` (0,5-15) y las coordenadas tienen escalas distintas y sin escalado la distancia euclídea ignoraría el ingreso.
- El cluster 2 (453 muestras en train) corresponde a zonas de ingreso muy alto incrustadas geográficamente.
- El `scaler` se guarda junto a los modelos: sin él, un punto nuevo se escalaría mal y caería en el cluster equivocado.

## Base

Proyecto construido sobre la [plantilla de ML de 4Geeks Academy](https://github.com/4GeeksAcademy/machine-learning-python-template).
