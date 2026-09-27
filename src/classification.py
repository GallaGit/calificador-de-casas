"""Fase 4: clasificación supervisada sobre las etiquetas generadas por K-Means.

X = [Latitude, Longitude, MedInc] (escaladas con el mismo scaler del K-Means).
y = cluster (0-5).
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "knn_k15": KNeighborsClassifier(n_neighbors=15),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
    }


def train_and_evaluate(
    X_train, y_train, X_test, y_test, random_state: int = 42
) -> tuple[str, object, dict]:
    """Entrena los 3 candidatos, evalúa en test y devuelve (mejor_nombre, mejor_modelo, resultados)."""
    results = {}
    best_name, best_model, best_acc = None, None, -1.0
    for name, model in get_models(random_state).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = {
            "accuracy": float(acc),
            "report": classification_report(y_test, pred, output_dict=True),
            "report_text": classification_report(y_test, pred),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "model": model,
        }
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc
    return best_name, best_model, results


def features_and_target(labeled_df: pd.DataFrame):
    X = labeled_df[["Latitude", "Longitude", "MedInc"]]
    y = labeled_df["cluster"].astype(int)
    return X, y
