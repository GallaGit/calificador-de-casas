# Fase 4 — Clasificación supervisada sobre los clusters

## Objetivo de la fase
Entrenar un clasificador que aprenda `X=[Latitude, Longitude, MedInc] → y=cluster` usando las etiquetas del K-Means, comparar candidatos y reportar métricas en test.

## Utilidad en el flujo
- Es el patrón auto-etiquetado → supervisado: el K-Means etiqueta barato y el clasificador aprende fronteras rápidas de inferir, serializables y evaluables con accuracy/precision/recall.
- Permite clasificar casas nuevas sin calcular distancias a centroides y detectar si algún cluster es confundible (p. ej. el 2, minoritario).
- Resultados reales en test (4.128):
  - `logistic_regression`: accuracy 0,9879
  - `knn_k15`: accuracy 0,9898
  - `random_forest` (200 árboles): accuracy 0,9956 ← elegido
  - Todos con F1 ≥ 0,96 por clase; el cluster 2 tiene recall 0,98 aun con solo 112 muestras en test. Detalle completo en `reports/metrics.md`.

## Tecnología y por qué
- `LogisticRegression(max_iter=1000)`: baseline lineal, rápido e interpretable; rinde ~0,988 porque los clusters son casi linealmente separables en espacio escalado.
- `KNeighborsClassifier(n_neighbors=15)`: baseline no paramétrico coherente con la geometría de centroides; `k=15` suaviza el ruido frente a `k=5`.
- `RandomForestClassifier(n_estimators=200, n_jobs=-1)`: captura fronteras no lineales (enclaves de ingreso alto dentro de una zona) y gana (+0,6 pp). 200 árboles bastan; más solo aumenta el `.pkl` de 12 MB.
- `accuracy_score`, `classification_report`, `confusion_matrix`: accuracy para ranking + precision/recall/F1 por clase porque el cluster 2 está desbalanceado (2,7% de train).
- Entrada escalada con el mismo `scaler` del K-Means: mantiene coherencia de features entre ambos modelos.

## Salidas
- `src/classification.py::get_models`, `::train_and_evaluate`
- `reports/metrics.md` con los tres reports
