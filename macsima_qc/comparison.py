"""
comparison.py
-------------
Comparaison de segmentations automatiques (MACSiQView) vs segmentation manuelle (Fiji).

Utilise le test de Kolmogorov-Smirnov pour mesurer la distance entre distributions
de features morphologiques, et produit des visualisations (barplot KS, KDE).

Fonctions principales
---------------------
- load_manual(path)
- load_autos(paths)
- normalize_datasets(df_manual, df_autos)
- compute_ks_distances(df_manual, df_autos, legend)
- plot_ks_barplot(distances_df, legend, output_dir)
- plot_kde(df_manual, df_autos, legend, output_dir)
- run_comparison(manual_path, auto_paths, auto_names, output_dir)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

from .utils import MANUAL_COLUMNS, MANUAL_RENAME_MAP, AUTO_RENAME_MAP


def load_manual(path: str) -> pd.DataFrame:
    """
    Charge et renomme la segmentation manuelle Fiji.

    Parameters
    ----------
    path : str
        Chemin vers le CSV de segmentation manuelle.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes normalisées (MANUAL_COLUMNS).
    """
    df = pd.read_csv(path)
    df = df.rename(columns=MANUAL_RENAME_MAP)
    missing = [c for c in MANUAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier manuel : {missing}")
    return df[MANUAL_COLUMNS].copy()


def load_autos(paths: list[str]) -> list[pd.DataFrame]:
    """
    Charge et renomme plusieurs segmentations automatiques MACSiQView.

    Parameters
    ----------
    paths : list[str]
        Liste de chemins vers les CSV de segmentation automatique.

    Returns
    -------
    list[pd.DataFrame]
        Liste de DataFrames avec les colonnes normalisées (MANUAL_COLUMNS).
    """
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.rename(columns=AUTO_RENAME_MAP)
        missing = [c for c in MANUAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans {path} : {missing}")
        dfs.append(df[MANUAL_COLUMNS].copy())
    return dfs


def normalize_datasets(
    df_manual: pd.DataFrame, df_autos: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Normalise tous les datasets via MinMaxScaler.
    Le scaler est fitté uniquement sur df_manual (référence).

    Parameters
    ----------
    df_manual : pd.DataFrame
    df_autos : list[pd.DataFrame]

    Returns
    -------
    tuple[pd.DataFrame, list[pd.DataFrame]]
        (df_manual_scaled, df_autos_scaled)
    """
    scaler = MinMaxScaler()
    df_manual = df_manual.copy()
    df_manual[MANUAL_COLUMNS] = scaler.fit_transform(df_manual[MANUAL_COLUMNS])

    df_autos_scaled = []
    for df in df_autos:
        df = df.copy()
        df[MANUAL_COLUMNS] = scaler.transform(df[MANUAL_COLUMNS])
        df_autos_scaled.append(df)

    return df_manual, df_autos_scaled


def compute_ks_distances(
    df_manual: pd.DataFrame,
    df_autos: list[pd.DataFrame],
    legend: dict,
) -> pd.DataFrame:
    """
    Calcule les distances KS moyennes entre la segmentation manuelle et chaque auto.

    Parameters
    ----------
    df_manual : pd.DataFrame
    df_autos : list[pd.DataFrame]
    legend : dict
        Mapping {AutoN: nom_lisible}, ex: {"Auto1": "Mask 3 - Single Cell"}

    Returns
    -------
    pd.DataFrame
        DataFrame avec une ligne par segmentation auto et une colonne par feature + 'Mean Distance'.
    """
    distances = {
        col: [ks_2samp(df_manual[col], df_auto[col])[0] for df_auto in df_autos]
        for col in MANUAL_COLUMNS
    }
    distances_df = pd.DataFrame(distances, index=list(legend.keys()))
    distances_df["Mean Distance"] = distances_df.mean(axis=1)

    best = distances_df["Mean Distance"].idxmin()
    print(f"✅ Segmentation la plus proche de la référence : {legend[best]}")
    return distances_df


def plot_ks_barplot(
    distances_df: pd.DataFrame,
    legend: dict,
    output_dir: str = "figures",
) -> None:
    """
    Génère et sauvegarde le barplot des distances KS moyennes (dark theme).

    Parameters
    ----------
    distances_df : pd.DataFrame
        Résultat de compute_ks_distances().
    legend : dict
        Mapping {AutoN: nom_lisible}.
    output_dir : str
        Dossier de sortie pour la figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    distances_df["Mean Distance"].plot(kind="bar", color="#E9EDC9", edgecolor="white", ax=ax)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="center", xytext=(0, 10),
            textcoords="offset points", color="white",
        )

    ax.set_title("Average Kolmogorov-Smirnov Distances by Segmentation", color="white")
    ax.set_ylabel("Average Distance", color="white")
    ax.set_xlabel("Segmentations", color="white")
    ax.set_xticks(range(len(legend)))
    ax.set_xticklabels(
        [legend[k] for k in distances_df.index], rotation=45, ha="right", color="white"
    )
    ax.tick_params(colors="white")
    ax.grid(True, linestyle="--", alpha=0.6, color="white")
    ax.set_facecolor("black")
    fig.set_facecolor("black")

    plt.tight_layout()
    out = os.path.join(output_dir, "ks_average_distances.png")
    plt.savefig(out, format="png", facecolor="black")
    plt.show()
    print(f"💾 Figure sauvegardée : {out}")


def plot_kde(
    df_manual: pd.DataFrame,
    df_autos: list[pd.DataFrame],
    legend: dict,
    output_dir: str = "figures",
    params: list[str] = None,
) -> None:
    """
    Génère et sauvegarde les courbes KDE comparatives (normalisation indépendante).

    Parameters
    ----------
    df_manual : pd.DataFrame
    df_autos : list[pd.DataFrame]
    legend : dict
    output_dir : str
    params : list[str]
        Features à afficher. Par défaut : ['area', 'perimeter', 'feret', 'mean_intensity'].
    """
    if params is None:
        params = ["area", "perimeter", "feret", "mean_intensity"]

    os.makedirs(output_dir, exist_ok=True)

    # Normalisation indépendante pour le KDE
    df_manual_kde = pd.DataFrame(
        MinMaxScaler().fit_transform(df_manual[MANUAL_COLUMNS]),
        columns=MANUAL_COLUMNS,
    )
    df_autos_kde = [
        pd.DataFrame(MinMaxScaler().fit_transform(df[MANUAL_COLUMNS]), columns=MANUAL_COLUMNS)
        for df in df_autos
    ]

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        ax.set_facecolor("white")
        for i, (key, label) in enumerate(legend.items()):
            sns.kdeplot(df_autos_kde[i][param], label=label, linewidth=2, ax=ax)
        sns.kdeplot(
            df_manual_kde[param],
            label="Manual reference",
            color="cyan",
            linestyle="--",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"Comparative Distribution - {param}")
        ax.set_xlabel("Normalized Value")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Segmentation", loc="upper center", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = os.path.join(output_dir, "kde_comparative_distributions.png")
    fig.savefig(out, format="png", facecolor="white")
    plt.show()
    print(f"💾 Figure sauvegardée : {out}")


def run_comparison(
    manual_path: str,
    auto_paths: list[str],
    auto_names: list[str] = None,
    output_dir: str = "figures",
) -> pd.DataFrame:
    """
    Pipeline complet de comparaison : charge, normalise, calcule KS, génère les figures.

    Parameters
    ----------
    manual_path : str
        Chemin vers le CSV de segmentation manuelle.
    auto_paths : list[str]
        Liste de chemins vers les CSV de segmentations automatiques.
    auto_names : list[str]
        Noms lisibles pour chaque segmentation auto (même ordre que auto_paths).
        Par défaut : ["Auto1", "Auto2", ...].
    output_dir : str
        Dossier de sortie pour les figures.

    Returns
    -------
    pd.DataFrame
        Tableau des distances KS.
    """
    if auto_names is None:
        auto_names = [f"Auto{i+1}" for i in range(len(auto_paths))]

    legend = {f"Auto{i+1}": name for i, name in enumerate(auto_names)}

    df_manual = load_manual(manual_path)
    df_autos = load_autos(auto_paths)
    df_manual_n, df_autos_n = normalize_datasets(df_manual, df_autos)

    distances_df = compute_ks_distances(df_manual_n, df_autos_n, legend)
    plot_ks_barplot(distances_df, legend, output_dir)
    plot_kde(df_manual_n, df_autos_n, legend, output_dir)

    return distances_df
