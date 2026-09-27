# Fase 5 — Guardado de modelos

## Objetivo de la fase
Persistir `scaler`, `kmeans` y el mejor clasificador en `models/` para reutilizarlos sin reentrenar.

## Utilidad en el flujo
- Separa entrenamiento de inferencia: cualquier script/notebook puede cargar los `.pkl` y clasificar casas nuevas.
- Fija el `scaler` junto a los modelos: sin él, un punto nuevo se escalaría mal y caería en el cluster equivocado.

## Tecnología y por qué
- `joblib.dump/load`: estándar de scikit-learn para objetos con arrays numpy, más eficiente que `pickle` puro y carga el `RandomForest` (12 MB) sin problema. Alternativa `pickle` descartada por peor rendimiento con numpy; ONNX descartado por innecesario para este tamaño.
- Ficheros:
  - `models/scaler.pkl` (927 B) — `StandardScaler` ajustado en train.
  - `models/kmeans.pkl` (66 KB) — `KMeans(k=6)`.
  - `models/classifier.pkl` (12 MB) — `RandomForest` ganador.
  - `models/metadata.json` — mejor modelo, accuracies, k, features y tamaños para auditoría.
- `models/` es la carpeta que ya trae el template; se respeta en lugar de crear otra.

## Cómo reutilizar
```python
import joblib
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")
clf = joblib.load("models/classifier.pkl")
x_scaled = scaler.transform([[34.05, -118.25, 5.0]])  # Lat, Lon, MedInc
print("cluster kmeans:", kmeans.predict(x_scaled))
print("cluster clasificador:", clf.predict(x_scaled))
```
