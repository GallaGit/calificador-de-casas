"""Fases 2 y 3: K-Means (k=6) + predicción sobre test."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def fit_kmeans(
    train_df: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42,
) -> tuple[StandardScaler, KMeans, pd.DataFrame]:
    """Ajusta scaler + KMeans solo con train y devuelve train con columna `cluster` (int)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_train_scaled)
    train_labeled = train_df.copy()
    train_labeled["cluster"] = labels.astype(int)
    return scaler, kmeans, train_labeled


def predict_clusters(
    scaler: StandardScaler, kmeans: KMeans, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Aplica el scaler ajustado y predice el cluster de cada punto de test."""
    X_test_scaled = scaler.transform(test_df)
    labels = kmeans.predict(X_test_scaled)
    test_labeled = test_df.copy()
    test_labeled["cluster"] = labels.astype(int)
    return test_labeled
