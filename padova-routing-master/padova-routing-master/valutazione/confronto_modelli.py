"""
valutazione/confronto_modelli.py
===================================
Confronto a runtime tra diverse versioni del modello ML (standard, loss
custom, anelli, 6 anelli, regionale...), sulle stesse coppie source/target,
misurando la riduzione % di nodi esplorati da Dijkstra sul grafo sanato rispetto a Dijkstra
vanilla (calcolato una sola volta, indipendente dal modello).
"""

import networkx as nx
import numpy as np
import pandas as pd

from src.algoritmi import dijkstra_con_nodi_visitati
from src.bcf import esegui_bcf, esporta_per_bcf
from src.grafo import costruisci_archi_ridotti, sanifica_grafo
from src.predizioni import genera_predizioni


def confronta_modelli_runtime(
    G: nx.MultiDiGraph,
    modelli: dict[str, object],
    coppie: list[tuple[str, object, object]],
    bcf_bin: str,
    bcf_input_path: str,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
) -> pd.DataFrame:
    """
    Per ciascun modello in `modelli` ({nome: oggetto_modello}), esegue
    l'intera pipeline su tutte le coppie e calcola la riduzione % di nodi
    esplorati da Dijkstra sul grafo sanato rispetto a Dijkstra vanilla.

    NOTA: Dijkstra vanilla viene calcolato una sola volta per coppia
    (indipendente dal modello), non ripetuto per ciascun modello.

    Restituisce un DataFrame (righe = coppie, colonne = nomi modello,
    + riga "Media" finale), pronto per tabella e grafico comparativo.
    """
    nodi_esplorati_baseline = {}
    for nome, source, target in coppie:
        visitati = dijkstra_con_nodi_visitati(G, source, target, weight=weight_attr)
        nodi_esplorati_baseline[nome] = len(visitati)
        print(f"Dijkstra vanilla — {nome}: {len(visitati)} nodi esplorati")
    print()

    risultati = {nome_modello: {} for nome_modello in modelli}

    for nome_modello, modello_corrente in modelli.items():
        print(f"=== Modello: {nome_modello} ===")

        for nome, source, target in coppie:
            try:
                y_hat, y_hat_int = genera_predizioni(
                    G, modello_corrente, target, scale_factor=scale_factor
                )
                archi, nodo_to_idx, art_idx, _ = costruisci_archi_ridotti(
                    G, y_hat_int, weight_attr=weight_attr
                )
                esporta_per_bcf(archi, art_idx, bcf_input_path)
                phi, _ = esegui_bcf(bcf_bin, bcf_input_path, art_idx, len(G.nodes()))
                G_san = sanifica_grafo(
                    G, y_hat_int, phi, nodo_to_idx, weight_attr=weight_attr
                )

                visitati_sanato = dijkstra_con_nodi_visitati(
                    G_san, source, target, weight=weight_attr
                )
                n_sanato = len(visitati_sanato)
                n_baseline = nodi_esplorati_baseline[nome]
                riduzione_pct = (1 - n_sanato / n_baseline) * 100 if n_baseline > 0 else 0

                risultati[nome_modello][nome] = riduzione_pct
                print(
                    f"  {nome}: {n_sanato} nodi (baseline {n_baseline})  "
                    f"→  {riduzione_pct:+.1f}%"
                )

            except Exception as ex:
                print(f"  ❌ {nome}: {ex}")
                risultati[nome_modello][nome] = None
        print()

    df = pd.DataFrame(risultati)
    nomi_coppie_ordine = [c[0] for c in coppie]
    df = df.loc[nomi_coppie_ordine]
    df.loc["Media"] = df.mean()

    return df


def plot_confronto_modelli(df_confronto: pd.DataFrame, output_path: str = "confronto_modelli.png"):
    """
    Grafico a barre raggruppate per confronta_modelli_runtime: una barra
    per modello, raggruppata per coppia, valori positivi = miglioramento.
    """
    import matplotlib.pyplot as plt

    df_plot = df_confronto.drop(index="Media", errors="ignore")
    coppie_nomi = list(df_plot.index)
    modelli_nomi = list(df_plot.columns)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(coppie_nomi))
    width = 0.8 / max(len(modelli_nomi), 1)
    colori = plt.cm.Set2(np.linspace(0, 1, len(modelli_nomi)))

    for j, (modello_nome, colore) in enumerate(zip(modelli_nomi, colori)):
        valori = df_plot[modello_nome].values
        ax.bar(x + j * width, valori, width, label=modello_nome, color=colore)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(modelli_nomi) - 1) / 2)
    ax.set_xticklabels(coppie_nomi, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Riduzione % nodi esplorati\n(Sanato vs Dijkstra vanilla)")
    ax.set_title("Confronto tra versioni del modello ML")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Grafico salvato come '{output_path}'")
