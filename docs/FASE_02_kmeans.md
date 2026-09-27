# Fase 2 — K-Means con k=6 sobre train

## Objetivo de la fase
Agrupar las casas en 6 clusters a partir de `Latitude`, `Longitude`, `MedInc` y guardar la etiqueta en la columna nueva `cluster` (int 0-5). Graficar `Longitude` vs `Latitude` coloreado por cluster.

## Utilidad en el flujo
- Convierte datos no etiquetados en etiquetas pseudo-supervisadas: es el paso de auto-etiquetado típico cuando no hay clases manuales.
- El mapa de puntos valida visualmente que los clusters tienen sentido geográfico (norte / bahía / centro / sur de California) en lugar de ser ruido.
- Distribución real en train: cluster 5 = 5.188 (zona sur densa), 3 = 3.931 (norte), 1 = 3.380, 4 = 2.239, 0 = 1.321 (centro disperso), 2 = 453 (grupo pequeño de ingreso muy alto).

## Tecnología y por qué
- `sklearn.preprocessing.StandardScaler` (ajustado solo en train): imprescindible porque `MedInc` (0,5-15) y coordenadas (-124 a -114, 32-42) tienen escalas distintas; sin escalado la distancia euclídea la dominaría la longitud/latitud e ignoraría el ingreso. Se guarda el scaler para reusar en test y en el clasificador.
- `sklearn.cluster.KMeans(n_clusters=6, n_init=10, random_state=42)`: algoritmo pedido en el enunciado, rápido para 16k filas, `n_init=10` evita mínimos locales malos, `random_state` lo hace reproducible. Alternativas descartadas: DBSCAN (no permite fijar k=6 ni predecir puntos nuevos con `predict` de forma natural) y GMM (más costoso sin ganancia clara aquí).
- `pandas` columna `cluster` como `int`: formato compacto y compatible como `y` para la Fase 4.
- `matplotlib` scatter (`s=4, alpha=0.5, cmap=tab10`): 16k puntos se ven sin saturar; el mapa resultante reproduce la silueta de California. Ver `reports/figures/clusters_train.png`.

## Lo que se ve en el gráfico
- Banda rosa al norte, mezcla rosa/oliva en la bahía de San Francisco, azul en el valle central, verde/cian en el sur (LA/San Diego).
- El cluster 2 (n=453) aparece disperso: corresponde al centroide con `MedInc` escalado ≈ +3,44, es decir, zonas de ingreso muy alto incrustadas geográficamente.

## Salidas
- `src/clustering.py::fit_kmeans`, `src/visualization.py::scatter_clusters`
- `data/processed/train_labeled.csv`, `reports/figures/clusters_train.png`
