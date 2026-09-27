"""Gráficos de clusters: train solo y train+test."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scatter_clusters(df, title: str, save_path: str | Path, label: str = "cluster"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        df["Longitude"], df["Latitude"], c=df[label], cmap="tab10",
        s=4, alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="cluster")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return str(save_path)


def scatter_train_test(train_df, test_df, save_path: str | Path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(train_df["Longitude"], train_df["Latitude"],
               c=train_df["cluster"], cmap="tab10", s=4, alpha=0.35, label="train")
    sc = ax.scatter(test_df["Longitude"], test_df["Latitude"],
                    c=test_df["cluster"], cmap="tab10", s=10, alpha=0.9,
                    edgecolors="black", linewidths=0.2, label="test")
    plt.colorbar(sc, ax=ax, label="cluster")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Clusters K-Means (k=6): train + test predicho")
    ax.legend(markerscale=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return str(save_path)
