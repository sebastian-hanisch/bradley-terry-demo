"""Existiert der Maximum-Likelihood-Schätzer? Ford-Bedingung (1957) in der Fassung für den Gewinngraphen: Der Schätzer existiert genau dann, wenn der
gerichtete Graph „i → j, wenn i mindestens einmal gegen j gewonnen hat“ stark zusammenhängend ist. Ein Ungeschlagener (kein eingehender Pfeil) oder ein
Siegloser (kein ausgehender) verletzt das; ebenso ein zerfallender Spielplan. Nur numpy/Standardbibliothek; der Test prüft gegen networkx."""

import numpy as np


def win_graph_components(W) -> list[tuple]:
    """Starke Zusammenhangskomponenten des Gewinngraphen (Kosaraju, iterativ), in Dominanzordnung: Eine Komponente steht vor jeder, die sie schlägt
    (topologische Ordnung der Kondensation, Quellen zuerst). Jede Komponente ist eine aufsteigend sortierte Tupel von Indizes."""
    A = np.asarray(W) > 0
    n = len(A)
    adj = [[int(j) for j in np.nonzero(A[i])[0] if j != i] for i in range(n)]
    radj = [[int(j) for j in np.nonzero(A[:, i])[0] if j != i] for i in range(n)]
    seen = [False] * n
    order: list[int] = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, iter(adj[s]))]
        while stack:
            v, it = stack[-1]
            for u in it:
                if not seen[u]:
                    seen[u] = True
                    stack.append((u, iter(adj[u])))
                    break
            else:
                order.append(v)
                stack.pop()
    comp = [-1] * n
    comps: list[tuple] = []
    for s in reversed(order):
        if comp[s] != -1:
            continue
        comp[s] = len(comps)
        members, stack2 = [], [s]
        while stack2:
            v = stack2.pop()
            members.append(v)
            for u in radj[v]:
                if comp[u] == -1:
                    comp[u] = len(comps)
                    stack2.append(u)
        comps.append(tuple(sorted(members)))
    return comps


def ford_condition(W) -> bool:
    """Wahr genau dann, wenn der Gewinngraph stark zusammenhängend ist (eine einzige Komponente); dann existiert der ungeschützte Schätzer endlich."""
    return len(win_graph_components(W)) == 1


def undefeated(W) -> list[int]:
    """Teilnehmer mit mindestens einer Partie und ohne Niederlage."""
    W = np.asarray(W)
    games = (W + W.T).sum(axis=1)
    losses = W.sum(axis=0)
    return [int(i) for i in np.nonzero((games > 0) & (losses == 0))[0]]


def winless(W) -> list[int]:
    """Teilnehmer mit mindestens einer Partie und ohne Sieg."""
    W = np.asarray(W)
    games = (W + W.T).sum(axis=1)
    wins = W.sum(axis=1)
    return [int(i) for i in np.nonzero((games > 0) & (wins == 0))[0]]
