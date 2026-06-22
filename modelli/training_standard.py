"""
modelli/training_standard.py
===============================
Modello baseline: HistGradientBoostingRegressor di scikit-learn, allenato
con loss MSE standard (nessuna penalità di consistenza). Usato come punto
di riferimento per misurare il beneficio della loss custom e dei modelli
ad anelli (vedi valutazione/confronto_modelli.py).
"""

import random

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.grafo import entro_raggio
from src.predizioni import _haversine_vettoriale


def genera_dataset_standard(
    G: nx.MultiDiGraph,
    centro_lat: float,
    centro_lon: float,
    raggio_km: float = 15.0,
    n_target: int = 60,
    n_nodi_per_target: int = 400,
    weight_attr: str = "travel_time_d",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un dataset di training filtrato a un raggio fisso dal centro
    (a differenza del modello ad anelli, qui non c'e' stratificazione per
    fascia: source e target sono campionati uniformemente entro raggio_km).

    Il filtro geografico e' importante: senza di esso, se il grafo G si
    estende per un'area molto più ampia (es. tutta la provincia/regione),
    il modello impara a discriminare bene scale larghe (facile, R² alto
    ma ingannevole) restando poco informativo per query a scala cittadina.
    """
    random.seed(seed)
    np.random.seed(seed)

    nodi_zona = entro_raggio(G, centro_lat, centro_lon, raggio_km)
    nodi_zona_set = set(nodi_zona)
    print(f"Nodi entro {raggio_km}km dal centro: {len(nodi_zona)} su {len(G.nodes())} totali")

    nodes_data = G.nodes(data=True)
    target_campionati = random.sample(nodi_zona, min(n_target, len(nodi_zona)))

    divisore = {
        "travel_time_s": 1.0, "travel_time_d": 10.0,
        "travel_time_c": 100.0, "travel_time_m": 1000.0,
    }.get(weight_attr, 10.0)

    righe_dataset = []
    G_rev = G.reverse(copy=False)
    for idx, target in enumerate(target_campionati):
        distanze = nx.single_source_dijkstra_path_length(G_rev, target, weight=weight_attr)
        target_lat, target_lon = nodes_data[target]["y"], nodes_data[target]["x"]

        nodi_raggiunti = [n for n in distanze.keys() if n in nodi_zona_set]
        nodi_campione = random.sample(
            nodi_raggiunti, min(n_nodi_per_target, len(nodi_raggiunti))
        )

        for nodo in nodi_campione:
            node_lat, node_lon = nodes_data[nodo]["y"], nodes_data[nodo]["x"]
            haversine_dist_m = _haversine_vettoriale(
                np.array([node_lat]), np.array([node_lon]), target_lat, target_lon
            )[0]
            tempo_reale_s = distanze[nodo] / divisore

            righe_dataset.append(
                {
                    "node_id": nodo,
                    "target_id": target,
                    "node_lat": node_lat,
                    "node_lon": node_lon,
                    "target_lat": target_lat,
                    "target_lon": target_lon,
                    "haversine_dist_m": haversine_dist_m,
                    "tempo_reale_s": tempo_reale_s,
                }
            )

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(target_campionati)} target processati...")

    df_train = pd.DataFrame(righe_dataset)
    print(f"\nDataset generato: {len(df_train)} esempi da {len(target_campionati)} target.")
    return df_train


def allena_modello_standard(
    df_train: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    max_iter: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 8,
):
    """
    Allena HistGradientBoostingRegressor con loss MSE standard (di default
    in scikit-learn). Nessuna penalità di consistenza: usato come baseline
    di confronto per i modelli con loss custom.
    """
    feature_cols = ["node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"]
    X = df_train[feature_cols]
    y = df_train["tempo_reale_s"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    print("\nTraining HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metriche = {
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
    }

    print("\n=== Valutazione modello ===")
    print(f"  MAE  train: {metriche['mae_train']:.1f}s   |  MAE  test: {metriche['mae_test']:.1f}s")
    print(f"  R²   train: {metriche['r2_train']:.3f}     |  R²   test: {metriche['r2_test']:.3f}")

    if metriche["mae_test"] > metriche["mae_train"] * 1.5:
        print("  ⚠️  MAE test molto più alto del train: possibile overfitting.")
    else:
        print("  ✅ MAE train/test comparabili: il modello generalizza ragionevolmente.")

    return model, feature_cols, metriche
