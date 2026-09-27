"""Fase 1: carga y split train/test del California Housing.

Solo usa las columnas Latitude, Longitude y MedInc.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Latitude", "Longitude", "MedInc"]
DEFAULT_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv"


def load_housing(path: str | Path, url: str = DEFAULT_URL) -> pd.DataFrame:
    """Carga housing.csv local; si no existe, lo descarga desde la URL del enunciado."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing}. Columnas: {list(df.columns)}")
    return df[FEATURES].copy()


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split estratificado no posible (sin etiquetas); split aleatorio reproducible."""
    return train_test_split(df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_housing("data/raw/housing.csv")
    train, test = split_data(df)
    print(f"Total: {df.shape} | train: {train.shape} | test: {test.shape}")
    print(train.describe().to_string())
