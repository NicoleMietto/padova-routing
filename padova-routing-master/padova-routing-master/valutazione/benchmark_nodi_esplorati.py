"""
valutazione/benchmark_nodi_esplorati.py
==========================================
Confronto tra Dijkstra vanilla e Dijkstra eseguito sul grafo sanato con i
potenziali predetti, sia su poche coppie manuali sia su larga scala con
campionamento stratificato per fascia geografica.

Metrica chiave: numero di nodi esplorati (non il tempo wall-clock, che e'
dominato da overhead di implementazione — vedi note di sviluppo). Una
riduzione % positiva significa che Dijkstra sul grafo sanato esplora meno nodi di Dijkstra
vanilla, cioe' le predizioni stanno effettivamente aiutando la ricerca.
"""

import random

import networkx as nx
import pandas as pd

from src.algoritmi import dijkstra_con_nodi_visitati
from src.bcf import esegui_bcf, esporta_per_bcf
from src.grafo import costruisci_archi_ridotti, sanifica_grafo
from src.predizioni import genera_predizioni


def confronta_nodi_esplorati(
    G: nx.MultiDiGraph,
    model,
    coppie: list[tuple[str, object, object]],
    bcf_bin: str,
    bcf_input_path: str,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
) -> pd.DataFrame:
    """
    Per ciascuna coppia (nome, source, target) in `coppie`, esegue l'intera
    pipeline (predizioni -> BCF -> sanazione -> Dijkstra) e confronta i nodi
    esplorati con Dijkstra vanilla.

    Restituisce un DataFrame con una riga per coppia.
    """
    risultati = []

    for nome, source, target in coppie:
        try:
            n_baseline = len(
                dijkstra_con_nodi_visitati(G, source, target, weight=weight_attr)
            )

            y_hat, y_hat_int = genera_predizioni(G, model, target, scale_factor=scale_factor)
            archi, nodo_to_idx, art_idx, n_neg = costruisci_archi_ridotti(
                G, y_hat_int, weight_attr=weight_attr
            )
            esporta_per_bcf(archi, art_idx, bcf_input_path)
            phi, _ = esegui_bcf(bcf_bin, bcf_input_path, art_idx, len(G.nodes()))
            G_san = sanifica_grafo(G, y_hat_int, phi, nodo_to_idx, weight_attr=weight_attr)

            n_sanato = len(
                dijkstra_con_nodi_visitati(G_san, source, target, weight=weight_attr)
            )
            riduzione_pct = (1 - n_sanato / n_baseline) * 100 if n_baseline > 0 else 0

            print(
                f"{nome}: Dijkstra={n_baseline}, Sanato={n_sanato}  "
                f"({riduzione_pct:+.1f}%)"
            )

            risultati.append(
                {
                    "coppia": nome, "nodi_baseline": n_baseline,
                    "nodi_sanato": n_sanato, "riduzione_pct": riduzione_pct,
                    "n_negativi": n_neg, "trovato": True,
                }
            )
        except Exception as ex:
            print(f"❌ {nome}: {ex}")
            risultati.append({"coppia": nome, "trovato": False, "errore": str(ex)})

    return pd.DataFrame(risultati)


def genera_coppie_stratificate(
    nodi_per_fascia: dict[int, list],
    nomi_fasce: list[str],
    tutti_nodi: list,
    n_coppie_per_fascia: int = 100,
    seed: int = 123,
) -> list[tuple[str, object, object, str]]:
    """
    Genera coppie (nome, source, target, fascia) casuali, stratificate per
    fascia del TARGET (il source può essere ovunque nel grafo, simulando
    query realistiche "qualcuno, da qualche parte, verso quella zona").

    Restituisce una lista di tuple a 4 elementi (a differenza delle coppie
    "manuali" a 3 elementi usate altrove, qui si porta anche l'etichetta
    di fascia per l'aggregazione successiva).
    """
    random.seed(seed)
    coppie = []

    for fascia_idx, nome_fascia in enumerate(nomi_fasce):
        nodi_disponibili = nodi_per_fascia[fascia_idx]
        n_generate, tentativi = 0, 0

        while n_generate < n_coppie_per_fascia and tentativi < n_coppie_per_fascia * 5:
            tentativi += 1
            target = random.choice(nodi_disponibili)
            source = random.choice(tutti_nodi)
            if source == target:
                continue
            coppie.append((f"{nome_fascia} #{n_generate + 1}", source, target, nome_fascia))
            n_generate += 1

        if n_generate < n_coppie_per_fascia:
            print(
                f"⚠️  Generate solo {n_generate}/{n_coppie_per_fascia} coppie "
                f"per {nome_fascia} dopo {tentativi} tentativi."
            )

    return coppie


def confronta_nodi_esplorati_stratificato(
    G: nx.MultiDiGraph,
    model,
    coppie_stratificate: list[tuple[str, object, object, str]],
    bcf_bin: str,
    bcf_input_path: str,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
    progress_ogni: int = 50,
) -> pd.DataFrame:
    """
    Versione di confronta_nodi_esplorati che accetta coppie con etichetta
    di fascia (vedi genera_coppie_stratificate) e la riporta nel risultato,
    per permettere l'aggregazione per fascia geografica.
    """
    risultati = []

    for i, (nome, source, target, fascia) in enumerate(coppie_stratificate):
        try:
            n_baseline = len(
                dijkstra_con_nodi_visitati(G, source, target, weight=weight_attr)
            )

            y_hat, y_hat_int = genera_predizioni(G, model, target, scale_factor=scale_factor)
            archi, nodo_to_idx, art_idx, _ = costruisci_archi_ridotti(
                G, y_hat_int, weight_attr=weight_attr
            )
            esporta_per_bcf(archi, art_idx, bcf_input_path)
            phi, _ = esegui_bcf(bcf_bin, bcf_input_path, art_idx, len(G.nodes()))
            G_san = sanifica_grafo(G, y_hat_int, phi, nodo_to_idx, weight_attr=weight_attr)

            n_sanato = len(
                dijkstra_con_nodi_visitati(G_san, source, target, weight=weight_attr)
            )
            riduzione_pct = (1 - n_sanato / n_baseline) * 100 if n_baseline > 0 else 0

            risultati.append(
                {
                    "coppia": nome, "fascia": fascia,
                    "nodi_baseline": n_baseline, "nodi_sanato": n_sanato,
                    "riduzione_pct": riduzione_pct, "trovato": True,
                }
            )
        except Exception as ex:
            risultati.append(
                {"coppia": nome, "fascia": fascia, "trovato": False, "errore": str(ex)}
            )

        if (i + 1) % progress_ogni == 0:
            print(f"  {i + 1}/{len(coppie_stratificate)} coppie processate...")

    df = pd.DataFrame(risultati)
    n_falliti = (~df["trovato"]).sum()
    if n_falliti > 0:
        print(f"\n⚠️  {n_falliti} coppie fallite (escluse dall'analisi).")

    return df


def aggrega_per_fascia(df_risultati: pd.DataFrame, nomi_fasce: list[str]) -> pd.DataFrame:
    """
    Aggrega i risultati di confronta_nodi_esplorati_stratificato per fascia,
    calcolando media, mediana, deviazione standard e percentuale di coppie
    con riduzione positiva (la metrica più robusta a outlier).
    """
    df_ok = df_risultati[df_risultati["trovato"]]

    agg = df_ok.groupby("fascia")["riduzione_pct"].agg(
        media="mean", mediana="median", std="std",
        pct_positive=lambda x: (x > 0).mean() * 100,
        n="count",
    )
    agg = agg.loc[[f for f in nomi_fasce if f in agg.index]]

    print("\n=== RISULTATI AGGREGATI PER FASCIA ===\n")
    print(agg.round(1))

    media_globale = df_ok["riduzione_pct"].mean()
    pct_positive_globale = (df_ok["riduzione_pct"] > 0).mean() * 100
    print(f"\nMedia globale su {len(df_ok)} coppie: {media_globale:+.1f}%")
    print(f"Percentuale di coppie con riduzione positiva: {pct_positive_globale:.1f}%")

    return agg
