"""
src/interpolazione.py
======================
Implementazione della proposta "interpolazione spaziale" (dalla mail al
prof. Scquizzato): il modello ML predice le variabili duali solo per un
sottoinsieme di nodi "ancora" (campionati casualmente), e i restanti nodi
ottengono il loro potenziale per interpolazione spaziale (triangolazione
di Delaunay via scipy.interpolate.LinearNDInterpolator), con fallback al
vicino più prossimo per i nodi fuori dal convex hull delle ancore.

Vantaggio: il costo di model.predict() (il vero collo di bottiglia nella
generazione delle predizioni, vedi note di sviluppo) scala con
sample_ratio * N invece che con N, mentre l'interpolazione stessa è
quasi istantanea (la triangolazione di Delaunay e' calcolata in C da
scipy).

ATTENZIONE — segno delle predizioni: questa funzione usa
`y_hat_int = round(val * scale_factor)` (segno DIRETTO), mentre
predizioni.genera_predizioni usa `-y_arr * scale_factor` (segno
INVERTITO, validato come corretto in valutazione/consistenza.py).
Prima di usare questa funzione in produzione, verificarne il segno con
valutazione.consistenza.sanity_check_segno — potrebbe essere necessario
invertire il segno qui per coerenza con il resto della pipeline.
"""

import random

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def genera_predizioni_interpolate(
    G: nx.MultiDiGraph,
    model,
    target,
    sample_ratio: float = 0.1,
    scale_factor: float = 10.0,
    min_ancore: int = 100,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """
    Genera i potenziali predetti calcolando il modello ML solo su un
    campione di nodi "ancora" (sample_ratio della popolazione, minimo
    min_ancore), e ottenendo i restanti per interpolazione spaziale
    lineare (Delaunay) con fallback nearest-neighbor.

    Il target e' sempre incluso tra le ancore, per garantire che il suo
    potenziale resti esattamente 0 (come nella convenzione di
    predizioni.genera_predizioni) e non sia soggetto a errore di
    interpolazione.

    Restituisce (y_hat, y_hat_int), stessa struttura di
    predizioni.genera_predizioni — ma vedere l'avviso sul segno nel
    docstring del modulo prima di usarla come drop-in replacement.
    """
    if seed is not None:
        random.seed(seed)

    nodi_totali = list(G.nodes(data=True))
    N = len(nodi_totali)

    num_anchors = max(min_ancore, int(N * sample_ratio))
    tutti_ids = [n[0] for n in nodi_totali]
    anchor_ids = set(random.sample(tutti_ids, min(num_anchors, N)))
    anchor_ids.add(target)  # garantisce precisione esatta sul target

    anchors = [n for n in nodi_totali if n[0] in anchor_ids]
    others = [n for n in nodi_totali if n[0] not in anchor_ids]

    target_lat = G.nodes[target]["y"]
    target_lon = G.nodes[target]["x"]

    X_anchors, anchor_coords = [], []
    for n_id, data in anchors:
        node_lat, node_lon = data["y"], data["x"]
        anchor_coords.append([node_lon, node_lat])  # (x, y) per l'interpolatore
        hav_dist = ox.distance.great_circle(node_lat, node_lon, target_lat, target_lon)
        X_anchors.append([node_lat, node_lon, target_lat, target_lon, hav_dist])

    feature_cols = ["node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"]
    df_anchors = pd.DataFrame(X_anchors, columns=feature_cols)

    # Inferenza ML solo sulle ancore — il vero collo di bottiglia, ridotto
    # a sample_ratio * N invece di N
    y_anchors_raw = model.predict(df_anchors)

    if others:
        interp_lin = LinearNDInterpolator(anchor_coords, y_anchors_raw)
        interp_near = NearestNDInterpolator(anchor_coords, y_anchors_raw)

        other_coords = [[data["x"], data["y"]] for n_id, data in others]
        y_others_raw = interp_lin(other_coords)

        # I nodi fuori dal convex hull delle ancore restituiscono NaN da
        # LinearNDInterpolator: fallback al vicino più prossimo.
        nan_mask = np.isnan(y_others_raw)
        if np.any(nan_mask):
            nan_coords = np.array(other_coords)[nan_mask]
            y_others_raw[nan_mask] = interp_near(nan_coords)
    else:
        y_others_raw = np.array([])

    y_hat_raw, y_hat_int = {}, {}

    for i, (n_id, _) in enumerate(anchors):
        val = y_anchors_raw[i]
        y_hat_raw[n_id] = val
        y_hat_int[n_id] = int(round(val * scale_factor))

    for i, (n_id, _) in enumerate(others):
        val = y_others_raw[i]
        y_hat_raw[n_id] = val
        y_hat_int[n_id] = int(round(val * scale_factor))

    return y_hat_raw, y_hat_int
