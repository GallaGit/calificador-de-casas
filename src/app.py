"""Orquestador por fases: carga -> KMeans -> test -> supervisado -> guardado.

Uso:
    python3 src/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_housing, split_data
from clustering import fit_kmeans, predict_clusters
from classification import features_and_target, train_and_evaluate
from visualization import scatter_clusters, scatter_train_test

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "housing.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Fase 1
    df = load_housing(RAW)
    train, test = split_data(df)
    print(f"[Fase 1] total={df.shape} train={train.shape} test={test.shape}")

    # Fase 2
    scaler, kmeans, train_labeled = fit_kmeans(train)
    print("[Fase 2] centros (escalados):\n", kmeans.cluster_centers_)
    print(train_labeled["cluster"].value_counts().sort_index().to_string())
    train_labeled.to_csv(PROCESSED / "train_labeled.csv", index=False)
    scatter_clusters(train_labeled, "Clusters K-Means (k=6) sobre train",
                     FIGURES / "clusters_train.png")

    # Fase 3
    test_labeled = predict_clusters(scaler, kmeans, test)
    test_labeled.to_csv(PROCESSED / "test_labeled.csv", index=False)
    scatter_train_test(train_labeled, test_labeled, FIGURES / "clusters_train_test.png")
    print("[Fase 3] test predicho:", test_labeled["cluster"].value_counts().sort_index().to_dict())

    # Fase 4
    X_train_raw, y_train = features_and_target(train_labeled)
    X_test_raw, y_test = features_and_target(test_labeled)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    best_name, best_model, results = train_and_evaluate(X_train, y_train, X_test, y_test)
    for name, r in results.items():
        print(f"[Fase 4] {name}: accuracy={r['accuracy']:.4f}")
        print(r["report_text"])
    print(f"[Fase 4] Mejor modelo: {best_name}")

    # Fase 5
    joblib.dump(kmeans, MODELS / "kmeans.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(best_model, MODELS / "classifier.pkl")
    summary = {
        "best_model": best_name,
        "accuracy": {n: r["accuracy"] for n, r in results.items()},
        "kmeans_clusters": 6,
        "features": ["Latitude", "Longitude", "MedInc"],
        "train_size": len(train_labeled),
        "test_size": len(test_labeled),
    }
    (MODELS / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[Fase 5] modelos guardados en models/")

    # reports breve
    report = ["# Reporte de resultados\n"]
    for name, r in results.items():
        report.append(f"## {name} (accuracy={r['accuracy']:.4f})\n```\n{r['report_text']}\n```\n")
    (ROOT / "reports" / "metrics.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
