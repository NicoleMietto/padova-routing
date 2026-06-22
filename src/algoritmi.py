"""
src/algoritmi.py
=================
Algoritmi di ricerca del cammino minimo usati per il benchmark: Dijkstra
con tracciamento dei nodi esplorati (per confrontare Dijkstra vanilla con
A* = Dijkstra sul grafo sanato), e Bellman-Ford come alternativa pura
Python al motore C++ BCF.
"""

import heapq
import time

import networkx as nx


def dijkstra_con_nodi_visitati(
    G: nx.MultiDiGraph, source, target, weight: str = "travel_time_d"
) -> set:
    """
    Dijkstra con tracciamento esplicito dei nodi visitati (chiusi durante
    la ricerca), usato per misurare quanti nodi esplora effettivamente
    l'algoritmo — non solo se trova il percorso.

    Eseguito sul grafo originale = Dijkstra vanilla.
    Eseguito sul grafo sanato (dopo sanifica_grafo) = equivalente ad A*
    sul grafo originale con euristica pari ai potenziali predetti.

    Restituisce l'insieme dei nodi visitati (incluso il target, se
    raggiunto). len(risultato) e' la metrica chiave per il confronto.
    """
    queue = [(0, source)]
    dist = {source: 0}
    visited = set()

    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for _, v, key, data in G.edges(u, keys=True, data=True):
            if v in visited:
                continue
            costo_arco = data.get(weight, 1)
            nuova_dist = d + costo_arco
            if nuova_dist < dist.get(v, float("inf")):
                dist[v] = nuova_dist
                heapq.heappush(queue, (nuova_dist, v))

    return visited


def dijkstra_benchmark(
    G: nx.MultiDiGraph, source, target, weight: str = "travel_time_d"
) -> tuple[float, int]:
    """
    Variante di dijkstra_con_nodi_visitati che restituisce anche la
    distanza finale, oltre al conteggio dei nodi esplorati.

    Restituisce (distanza, numero_nodi_esplorati).
    """
    visited = dijkstra_con_nodi_visitati(G, source, target, weight=weight)
    # Ricalcola la distanza con networkx (più leggibile che tracciarla a mano)
    try:
        distanza = nx.shortest_path_length(G, source, target, weight=weight)
    except nx.NetworkXNoPath:
        distanza = float("inf")
    return distanza, len(visited)


def bellman_ford_python(archi: list, super_idx: int) -> tuple[dict, float]:
    """
    Bellman-Ford con early stopping, usato come alternativa pura Python al
    motore C++ BCF. Calcola le distanze dal super-nodo (super_idx) a tutti
    gli altri nodi, usando direttamente la lista di archi già costruita da
    grafo.costruisci_archi_ridotti (che include già gli archi del
    super-nodo verso tutti i nodi reali, con peso 0).

    Restituisce (phi: {idx -> distanza}, tempo_secondi).
    """
    edges, nodes = [], set()
    for linea in archi:
        parti = linea.split()
        if len(parti) == 3:
            u, v, w = int(parti[0]), int(parti[1]), int(parti[2])
            edges.append((u, v, w))
            nodes.update([u, v])

    dist = {n: float("inf") for n in nodes}
    dist[super_idx] = 0

    t0 = time.time()
    for i in range(len(nodes) - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            print(f"  BF stabilizzato all'iterazione {i + 1}/{len(nodes) - 1}")
            break

    phi = {k: v for k, v in dist.items() if k != super_idx and v != float("inf")}
    return phi, time.time() - t0
