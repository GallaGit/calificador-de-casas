# Fase 3 — Predicción de clusters sobre test

## Objetivo de la fase
Usar el `scaler` + `KMeans` ajustados en train para predecir el cluster de cada punto de test (`kmeans.predict`) y superponerlos al gráfico de train.

## Utilidad en el flujo
- Simula producción: llegan casas nuevas y el modelo debe asignarles región sin reentrenar.
- Si los puntos de test caen dentro de las manchas de su color, el clustering generaliza; si aparecieran colores mezclados al azar, habría sobreajuste o mala elección de k.
- Resultado real: test 4.128 → {0: 337, 1: 829, 2: 112, 3: 962, 4: 514, 5: 1.374}, proporciones casi idénticas a train, y en `reports/figures/clusters_train_test.png` cada punto de test (borde negro) cae sobre su mancha.

## Tecnología y por qué
- `StandardScaler.transform` (no `fit_transform`): reutiliza media/varianza de train para no filtrar información de test.
- `KMeans.predict`: asigna al centroide más cercano en espacio escalado; es O(k) por punto, apto para inferencia en tiempo real. Alternativa re-ajustar K-Means con todo el dataset descartada porque invalidaría la evaluación.
- `matplotlib` overlay con dos `scatter`: train tenue (`alpha=0.35`) + test opaco con borde negro para distinguirlos. Ver `src/visualization.py::scatter_train_test`.

## Salidas
- `src/clustering.py::predict_clusters`
- `data/processed/test_labeled.csv`, `reports/figures/clusters_train_test.png`
