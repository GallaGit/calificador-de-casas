# Documentación por fases — Calificador de casas

Pipeline California Housing: `Latitude`, `Longitude`, `MedInc` → K-Means (k=6) → clasificador supervisado.

- [Fase 1 — Carga y split](FASE_01_carga_y_split.md): utilidad del recorte a 3 columnas y del split; tech `pandas` + `train_test_split` y por qué.
- [Fase 2 — K-Means](FASE_02_kmeans.md): auto-etiquetado en 6 clusters; tech `StandardScaler` + `KMeans` y por qué; lectura del mapa.
- [Fase 3 — Predicción en test](FASE_03_prediccion_test.md): validación en puntos nuevos; tech `transform` + `predict` y por qué.
- [Fase 4 — Clasificación supervisada](FASE_04_clasificacion_supervisada.md): comparativa LogReg / KNN / RandomForest (ganador RF 0,9956); tech y métricas y por qué.
- [Fase 5 — Guardado](FASE_05_guardado_modelos.md): persistencia con `joblib` en `models/` y por qué.

Ejecución completa: `python3 src/app.py`. Salidas: `data/processed/`, `reports/figures/`, `reports/metrics.md`, `models/*.pkl`.
